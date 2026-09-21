import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_error(x, y, eps=1e-8):
    min_len = min(x.shape[-1], y.shape[-1])
    x = x[..., :min_len]
    y = y[..., :min_len]
    error = x - y
    variance = torch.var(error) + eps
    precision = 1.0 / variance
    return precision * error


class BioLayer(nn.Module):


    def __init__(self, in_dim, out_dim, rank=16, mix_factor=0.05,
                 smoothing_factor=0.1, bias=True):
        super().__init__()

        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(out_dim, rank) * 0.02)

        self.rank = rank
        self.mix_factor = mix_factor
        self.smoothing_factor = smoothing_factor

        self.register_buffer("stabiliser", torch.tensor(1.0))
        self.register_buffer("current_error", torch.zeros(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.last_x = None
        self.y_pred = None

    def forward(self, x):
        self.last_x = x
        out = x @ self.W
        if self.use_bias:
            out = out + self.bias
        self.y_pred = out
        return out

    @torch.no_grad()
    def local_update(self, y, lr_w=1e-3, lr_lr=1e-3):
        if self.last_x is None or self.y_pred is None:
            return

        if y.ndim == 1:
            y = y.unsqueeze(0)
        y_pred = self.y_pred if self.y_pred.ndim > 1 else self.y_pred.unsqueeze(0)
        last_x = self.last_x if self.last_x.ndim > 1 else self.last_x.unsqueeze(0)

        error = calculate_error(y_pred, y)
        self.current_error = error

        mean_err = error.mean()
        self.stabiliser = (1 - self.smoothing_factor) * self.stabiliser + \
            self.smoothing_factor * mean_err

        grad_W = last_x.mT @ error
        self.W.data.add_(grad_W, alpha=-lr_w)

        if self.use_bias:
            grad_bias = error.mean(0)
            self.bias.data.add_(grad_bias, alpha=-lr_w)

        E_r = error @ self.B
        grad_A = last_x.mT @ E_r
        self.A.data.add_(grad_A, alpha=-lr_lr)

        X_A = last_x @ self.A
        grad_B = error.mT @ X_A
        self.B.data.add_(grad_B, alpha=-lr_lr)

        lowrank_W = self.A @ self.B.T
        self.W.data.mul_(1 - self.mix_factor).add_(lowrank_W, alpha=self.mix_factor)


class ExternalMemory(nn.Module):


    def __init__(self, dim, num_slots=128, write_lr=0.1, decay=0.995,
                 max_norm=30.0):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.write_lr = write_lr
        self.decay = decay
        self.max_norm = max_norm

        memory_init = torch.randn(num_slots, dim) * 0.02
        self.register_buffer("memory", memory_init)

        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.write_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def read(self, x):
        q = self.key_proj(x)
        scores = torch.matmul(q, self.memory.t()) / math.sqrt(self.dim)
        attn = torch.softmax(scores, dim=-1)
        read_val = torch.matmul(attn, self.memory)
        return self.out_proj(read_val), attn

    @torch.no_grad()
    def write(self, x, attn):
        write_val = self.write_proj(x)
        B, T, D = write_val.shape
        S = self.num_slots

        attn_flat = attn.reshape(B * T, S)
        write_flat = write_val.reshape(B * T, D)

        usage = attn_flat.sum(dim=0).clamp(min=1e-6).unsqueeze(-1)
        update = (attn_flat.t() @ write_flat) / usage

        mem = self.decay * self.memory + self.write_lr * update
        slot_norm = mem.norm(dim=1, keepdim=True)
        self.memory = mem * (self.max_norm / slot_norm.clamp(min=self.max_norm))


def wkv_recurrence(k, v, decay):

    orig_dtype = k.dtype
    k = k.float()
    v = v.float()
    decay = decay.float()

    B, T, H, Hd = k.shape
    kv = k * v

    d = decay.view(1, 1, H, 1)
    t_idx = torch.arange(T, device=k.device, dtype=k.dtype).view(1, T, 1, 1)

    d_pows_i = torch.pow(d, t_idx)
    kv_scaled = kv / (d_pows_i + 1e-8)

    cs = torch.cumsum(kv_scaled, dim=1)
    d_pows_t = d_pows_i

    state = cs * d_pows_t
    return state.to(orig_dtype)


class RWKVBlock(nn.Module):
    def __init__(self, dim, heads=8, rank=16, ffn_mult=4, bias=False):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.in_dim = dim
        self.out_dim = dim
        self.heads = heads
        self.hdim = dim // heads

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.r = BioLayer(dim, dim, rank=rank, bias=bias)
        self.k = BioLayer(dim, dim, rank=rank, bias=bias)
        self.v = BioLayer(dim, dim, rank=rank, bias=bias)

        self.register_buffer("log_decay", torch.zeros(heads))

        self.out = BioLayer(dim, dim, rank=rank, bias=bias)
        self.ffn_up = BioLayer(dim, dim * ffn_mult, rank=rank, bias=bias)
        self.ffn_down = BioLayer(dim * ffn_mult, dim, rank=rank, bias=bias)

    def forward(self, x):
        B, T, D = x.shape
        H, Hd = self.heads, self.hdim

        h = self.ln1(x)
        r = torch.sigmoid(self.r(h))
        k = self.k(h)
        v = self.v(h)

        r = r.view(B, T, H, Hd)
        k_h = k.view(B, T, H, Hd)
        v_h = v.view(B, T, H, Hd)

        decay = torch.exp(-F.softplus(self.log_decay))
        wkv = wkv_recurrence(k_h, v_h, decay)

        y = r * wkv
        y = y.reshape(B, T, D)
        y = self.out(y)
        x = x + y

        h2 = self.ln2(x)
        ff = self.ffn_up(h2)
        ff = F.silu(ff)
        ff = self.ffn_down(ff)
        x = x + ff

        if self.training:
            self.update_decay(k_h, v_h)

        return x

    @torch.no_grad()
    def update_decay(self, k, v, lr=1e-3):
        hebb = torch.mean(k.detach() * v.detach(), dim=(0, 1, 3))
        self.log_decay += lr * hebb
        self.log_decay.clamp_(-2.0, 6.0)


class RWKVLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim=384, n_layers=6, heads=8, rank=16,
                 ffn_mult=4, num_mem_slots=128, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, dim)
        self.memory = ExternalMemory(dim, num_slots=num_mem_slots)
        self.blocks = nn.ModuleList([
            RWKVBlock(dim, heads=heads, rank=rank, ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.embed(idx)

        mem_read, attn = self.memory.read(x)
        x = x + mem_read

        for block in self.blocks:
            x = block(x)

        if self.training:
            self.memory.write(x.detach(), attn.detach())

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def num_params(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters()
                   if not trainable_only or p.requires_grad)


MODEL_CONFIG = dict(
    vocab_size=50257,
    dim=768,
    n_layers=12,
    heads=12,
    rank=32,
    ffn_mult=4,
    num_mem_slots=512,
    max_seq_len=1024,
)


if __name__ == "__main__":
    model = RWKVLanguageModel(**MODEL_CONFIG)
    print(f"{model.num_params() / 1_000_000:.2f}M params")

    x = torch.randint(0, MODEL_CONFIG["vocab_size"], (2, 64))
    logits = model(x)
    print("logits shape:", logits.shape)
