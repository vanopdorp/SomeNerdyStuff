import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from model import RWKVLanguageModel, MODEL_CONFIG
from data import build_token_blocks, BlockDataset

import torch_xla.core.xla_model as xm


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--output_path", type=str, default="checkpoints/rwkv_bio_tpu.pt")
    p.add_argument("-f", type=str, default="")
    return p.parse_args(argv)


def evaluate(model, val_loader, device, max_batches=20):
    model.eval()
    losses = []
    with torch.no_grad():
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


def main(argv=None):
    args = parse_args(argv)

    device = xm.xla_device()
    torch.manual_seed(1234)

    print("Tokenizing wikitext-103 (cached by HF datasets na eerste run)...")
    train_data = build_token_blocks(args.seq_len, "train", cache_dir=args.cache_dir)
    val_data = build_token_blocks(args.seq_len, "validation", cache_dir=args.cache_dir)

    train_ds = BlockDataset(train_data)
    val_ds = BlockDataset(val_data)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1,
    )

    model.train()
    step = 0
    running_loss = 0.0

    while step < args.max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            xm.optimizer_step(optimizer)
            scheduler.step()

            running_loss += loss.item()
            step += 1

            if step % args.log_every == 0:
                avg = running_loss / args.log_every
                print(f"step {step:6d} | train_loss {avg:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")
                running_loss = 0.0

            if step % args.eval_every == 0 or step == args.max_steps:
                val_loss = evaluate(model, val_loader, device)
                ppl = float(np.exp(val_loss)) if val_loss == val_loss else float("nan")
                print(f"step {step:6d} | val_loss {val_loss:.4f} | val_ppl {ppl:.2f}")

            if step % args.ckpt_every == 0 or step == args.max_steps:
                os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), args.output_path)
                print(f"saved checkpoint -> {args.output_path}")

            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
