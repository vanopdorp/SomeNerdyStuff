import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import RWKVLanguageModel, MODEL_CONFIG
from data import build_token_blocks, BlockDataset


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def train(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(1234)

    print(f"Tokenizing wikitext-103 (cached by HF datasets after the first run)...")
    train_data = build_token_blocks(args.seq_len, "train", cache_dir=args.cache_dir)
    val_data = build_token_blocks(args.seq_len, "validation", cache_dir=args.cache_dir)

    train_ds = BlockDataset(train_data)
    val_ds = BlockDataset(val_data)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
    )

    cfg = dict(MODEL_CONFIG)
    cfg["max_seq_len"] = args.seq_len
    model = RWKVLanguageModel(**cfg).to(device)

    print(f"model: {model.num_params() / 1e6:.2f}M params | device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
    )
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    print(f"steps_per_epoch={steps_per_epoch} epochs={args.epochs} "
          f"total_steps={total_steps}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1,
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def autocast_ctx():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return _nullcontext()

    def evaluate(max_batches=20):
        model.eval()
        losses = []
        with torch.no_grad(), autocast_ctx():
            for i, (x, y) in enumerate(val_loader):
                if i >= max_batches:
                    break
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                losses.append(loss.item())
        model.train()
        if not losses:
            return float("nan")
        return sum(losses) / len(losses)

    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    while step < total_steps and epoch < args.epochs:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with autocast_ctx():
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if not torch.isfinite(loss):
                print(f"step {step:6d} | skipping non-finite loss batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            step += 1

            if step % args.log_every == 0:
                avg = running_loss / args.log_every
                print(f"step {step:6d} | train_loss {avg:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")
                running_loss = 0.0

            if step % args.eval_every == 0 or step >= total_steps:
                val_loss = evaluate()
                ppl = float(np.exp(val_loss)) if val_loss == val_loss else float("nan")
                print(f"step {step:6d} | val_loss {val_loss:.4f} | val_ppl {ppl:.2f}")

            if step % args.ckpt_every == 0 or step >= total_steps:
                os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), args.output_path)
                print(f"saved checkpoint -> {args.output_path}")

            if step >= total_steps:
                break

        print(f"completed epoch {epoch + 1}/{args.epochs} (step {step}/{total_steps})")
        epoch += 1


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4, help="batch size (single device)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None, help="cuda/cpu; default: auto")
    p.add_argument("--amp", action="store_true", help="enable fp16 autocast on CUDA")
    p.add_argument("--epochs", type=int, default=3,
                   help="number of full passes over the train set")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--ckpt_every", type=int, default=200)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--output_path", type=str, default="checkpoints/rwkv_bio_50m.pt")
    p.add_argument("-f", type=str, default="")
    return p.parse_args(argv)


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()