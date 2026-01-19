#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Diagnostic scan for induction heads on repeated random sequences.

This mirrors the common literature setup:
- build a batch of repeated random token sequences (pattern_len repeated twice),
- run GPT-2 with output_attentions,
- rank heads by induction stripe attention (q -> q - period),
- rank heads by previous-token attention (q -> q - 1).

Layer/head indices in the report are 0-based to match the literature.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import GPT2Model, GPT2Tokenizer
except ImportError as exc:
    raise SystemExit("Requires torch + transformers (use `uv run --with torch --with transformers`).") from exc


def select_vocab_candidates(
    tokenizer,
    vocab_min: int,
    vocab_max: int,
    min_word_length: int,
    require_leading_space: bool,
) -> list[int]:
    candidates = []
    for tid in range(vocab_min, vocab_max):
        word = tokenizer.decode([tid])
        if len(word.strip()) <= min_word_length:
            continue
        if require_leading_space and not word.startswith(" "):
            continue
        candidates.append(tid)
    return candidates


def build_batch(
    rng: np.random.Generator,
    candidates: list[int],
    batch_size: int,
    pattern_len: int,
    seq_len: int,
) -> np.ndarray:
    if seq_len != 2 * pattern_len:
        raise ValueError("seq_len must equal 2 * pattern_len for repeated sequence diagnostic")
    if len(candidates) < pattern_len:
        raise ValueError("Not enough vocab candidates for requested pattern length")
    batch = np.zeros((batch_size, seq_len), dtype=np.int64)
    for idx in range(batch_size):
        pattern = rng.choice(candidates, size=pattern_len, replace=False)
        batch[idx] = np.tile(pattern, 2)
    return batch


def compute_stripe_stats(attn: torch.Tensor, period: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seq_len, _ = attn.shape
    q = torch.arange(period, seq_len, device=attn.device)
    k = q - period
    block = attn[:, :, q, :]
    k_index = k.view(1, 1, -1, 1).expand(batch, heads, -1, 1)
    stripe_vals = block.gather(dim=-1, index=k_index).squeeze(-1)
    stripe_mean = stripe_vals.mean(dim=(0, 2))
    max_vals = block.max(dim=-1).values
    stripe_top1 = (stripe_vals >= max_vals - 1e-12).float().mean(dim=(0, 2))
    return stripe_mean, stripe_top1


def compute_prev_stats(attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seq_len, _ = attn.shape
    q = torch.arange(1, seq_len, device=attn.device)
    k = q - 1
    block = attn[:, :, q, :]
    k_index = k.view(1, 1, -1, 1).expand(batch, heads, -1, 1)
    prev_vals = block.gather(dim=-1, index=k_index).squeeze(-1)
    prev_mean = prev_vals.mean(dim=(0, 2))
    max_vals = block.max(dim=-1).values
    prev_top1 = (prev_vals >= max_vals - 1e-12).float().mean(dim=(0, 2))
    return prev_mean, prev_top1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--pattern-len", type=int, default=25)
    parser.add_argument("--batch", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-min", type=int, default=1000)
    parser.add_argument("--vocab-max", type=int, default=5000)
    parser.add_argument("--min-word-length", type=int, default=4)
    parser.add_argument("--require-leading-space", action="store_true", default=True)
    parser.add_argument("--allow-no-leading-space", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("reports/induction_diagnostic.txt"))
    args = parser.parse_args()

    require_leading_space = args.require_leading_space and not args.allow_no_leading_space
    rng = np.random.default_rng(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2Model.from_pretrained(args.model, attn_implementation=args.attn_implementation)
    model.eval()
    model.to(args.device)

    candidates = select_vocab_candidates(
        tokenizer,
        vocab_min=args.vocab_min,
        vocab_max=args.vocab_max,
        min_word_length=args.min_word_length,
        require_leading_space=require_leading_space,
    )
    batch_tokens = build_batch(
        rng,
        candidates,
        batch_size=args.batch,
        pattern_len=args.pattern_len,
        seq_len=args.seq_len,
    )
    input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=args.device)
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True, use_cache=False)
    if outputs.attentions is None:
        raise SystemExit("Model did not return attention weights.")

    stripe_scores = []
    prev_scores = []
    for layer_idx, attn in enumerate(outputs.attentions):
        stripe_mean, stripe_top1 = compute_stripe_stats(attn, args.pattern_len)
        prev_mean, prev_top1 = compute_prev_stats(attn)
        for head_idx in range(attn.shape[1]):
            stripe_scores.append((float(stripe_mean[head_idx]), float(stripe_top1[head_idx]), layer_idx, head_idx))
            prev_scores.append((float(prev_mean[head_idx]), float(prev_top1[head_idx]), layer_idx, head_idx))

    stripe_scores.sort(key=lambda x: x[0], reverse=True)
    prev_scores.sort(key=lambda x: x[0], reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii") as f:
        f.write("Induction diagnostic (repeated random sequence)\n")
        f.write(f"model={args.model}\n")
        f.write(f"seq_len={args.seq_len} pattern_len={args.pattern_len} batch={args.batch}\n")
        f.write(
            f"vocab_range=[{args.vocab_min},{args.vocab_max}) min_word_length={args.min_word_length} "
            f"leading_space={require_leading_space}\n"
        )
        f.write("\nTop induction stripe heads:\n")
        for rank, (mean, top1, layer, head) in enumerate(stripe_scores[: args.top], start=1):
            f.write(f"{rank:02d} L{layer}H{head} stripeMean={mean:.6f} stripeTop1={top1:.3f}\n")
        f.write("\nTop previous-token heads:\n")
        for rank, (mean, top1, layer, head) in enumerate(prev_scores[: args.top], start=1):
            f.write(f"{rank:02d} L{layer}H{head} prevMean={mean:.6f} prevTop1={top1:.3f}\n")

    print(f"Wrote report to {args.output}")
    print("\nTop induction stripe heads:")
    for rank, (mean, top1, layer, head) in enumerate(stripe_scores[: args.top], start=1):
        print(f"{rank:02d} L{layer}H{head} stripeMean={mean:.6f} stripeTop1={top1:.3f}")
    print("\nTop previous-token heads:")
    for rank, (mean, top1, layer, head) in enumerate(prev_scores[: args.top], start=1):
        print(f"{rank:02d} L{layer}H{head} prevMean={mean:.6f} prevTop1={top1:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
