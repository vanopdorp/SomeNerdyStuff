
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model import RWKVLanguageModel, MODEL_CONFIG
from data import build_token_blocks, BlockDataset


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def is_main(rank):
    return rank == 0


def evaluate(model, val_loader, device, max_batches=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def main(args):
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(1234 + rank)

    if is_main(rank):
        print("Tokenizing wikitext-103 (cached by 🤗 datasets after the first run)...")
    dist.barrier()
    train_data = build_token_blocks(args.seq_len, "train", cache_dir=args.cache_dir)
    val_data = build_token_blocks(args.seq_len, "validation", cache_dir=args.cache_dir)
    dist.barrier()

    train_ds = BlockDataset(train_data)
    val_ds = BlockDataset(val_data)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        drop_last=True, num_workers=4, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
    )

    cfg = dict(MODEL_CONFIG)
    cfg["max_seq_len"] = args.seq_len
    model = RWKVLanguageModel(**cfg).to(device)

    if is_main(rank):
        print(f"model: {model.num_params() / 1e6:.2f}M params, world_size={world_size}")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1,
    )
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        train_sampler.set_epoch(step)
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            sync = (i + 1) % args.grad_accum_steps == 0
            ctx = model.no_sync() if (not sync and world_size > 1) else _nullcontext()
            with ctx:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss = loss / args.grad_accum_steps
                scaler.scale(loss).backward()

            running_loss += loss.item() * args.grad_accum_steps

            if not sync:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % args.log_every == 0 and is_main(rank):
                avg = running_loss / (args.log_every * args.grad_accum_steps)
                print(f"step {step:6d} | train_loss {avg:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")
                running_loss = 0.0

            if (step % args.eval_every == 0 or step == args.max_steps) and is_main(rank):
                val_loss = evaluate(model, val_loader, device)
                ppl = float(np.exp(val_loss)) if val_loss == val_loss else float("nan")
                print(f"step {step:6d} | val_loss {val_loss:.4f} | val_ppl {ppl:.2f}")

            if (step % args.ckpt_every == 0 or step == args.max_steps) and is_main(rank):
                os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
                torch.save(model.module.state_dict(), args.output_path)
                print(f"saved checkpoint -> {args.output_path}")

            if step >= args.max_steps:
                break

    dist.barrier()
    dist.destroy_process_group()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32, help="per-GPU batch size")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--output_path", type=str, default="checkpoints/rwkv_bio_50m_gpu.pt")
    p.add_argument("-f", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())