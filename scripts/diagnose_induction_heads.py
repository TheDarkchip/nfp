#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Diagnostic scan for induction heads on repeated random sequences.

This aligns with TransformerLens head detection:
- build a batch of repeated random token sequences (pattern_len repeated twice),
- run GPT-2 with output_attentions,
- score heads via the induction detection pattern (duplicate-token mask shifted right),
- score previous-token heads via the previous-token detection pattern.

Scores use the TransformerLens "mul" metric by default:
  score = sum(attn * detection_pattern) / sum(attn)

Layer/head indices in the report are 0-based to match the literature.
"""

from __future__ import annotations

import argparse
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


def previous_token_detection_pattern(tokens: torch.Tensor) -> torch.Tensor:
    seq_len = tokens.shape[-1]
    detection_pattern = torch.zeros((seq_len, seq_len), device=tokens.device)
    detection_pattern[1:, :-1] = torch.eye(seq_len - 1, device=tokens.device)
    return torch.tril(detection_pattern)


def duplicate_token_detection_pattern(tokens: torch.Tensor) -> torch.Tensor:
    tok = tokens.view(-1)
    eq_mask = tok.unsqueeze(0).eq(tok.unsqueeze(1))
    eq_mask.fill_diagonal_(False)
    return torch.tril(eq_mask.to(dtype=torch.float32))


def induction_detection_pattern(tokens: torch.Tensor) -> torch.Tensor:
    duplicate_pattern = duplicate_token_detection_pattern(tokens)
    shifted = torch.roll(duplicate_pattern, shifts=1, dims=1)
    zeros_column = torch.zeros((duplicate_pattern.shape[0], 1), device=duplicate_pattern.device)
    result = torch.cat((zeros_column, shifted[:, 1:]), dim=1)
    return torch.tril(result)


def compute_head_score(
    attention_pattern: torch.Tensor,
    detection_pattern: torch.Tensor,
    *,
    exclude_bos: bool,
    exclude_current_token: bool,
    error_measure: str,
) -> float:
    if error_measure == "mul":
        mask = torch.ones_like(attention_pattern)
        if exclude_bos:
            mask[:, 0] = 0
        if exclude_current_token:
            mask.fill_diagonal_(0)
        masked_attn = attention_pattern * mask
        denom = masked_attn.sum()
        if denom == 0:
            return 0.0
        return ((masked_attn * detection_pattern).sum() / denom).item()

    abs_diff = (attention_pattern - detection_pattern).abs()
    if exclude_bos:
        abs_diff[:, 0] = 0
    if exclude_current_token:
        abs_diff.fill_diagonal_(0)
    size = abs_diff.shape[0]
    return 1 - round((abs_diff.mean() * size).item(), 3)


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
    parser.add_argument("--exclude-bos", action="store_true")
    parser.add_argument("--exclude-current-token", action="store_true")
    parser.add_argument(
        "--error-measure",
        choices=("mul", "abs"),
        default="mul",
        help="TransformerLens-style scoring metric",
    )
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

    num_layers = len(outputs.attentions)
    num_heads = outputs.attentions[0].shape[1]
    induction_scores = np.zeros((num_layers, num_heads), dtype=np.float64)
    prev_scores = np.zeros((num_layers, num_heads), dtype=np.float64)

    for batch_idx in range(input_ids.shape[0]):
        tokens = input_ids[batch_idx]
        induction_pattern = induction_detection_pattern(tokens)
        prev_pattern = previous_token_detection_pattern(tokens)
        for layer_idx, attn in enumerate(outputs.attentions):
            layer_attn = attn[batch_idx]
            for head_idx in range(layer_attn.shape[0]):
                head_attn = layer_attn[head_idx]
                induction_scores[layer_idx, head_idx] += compute_head_score(
                    head_attn,
                    detection_pattern=induction_pattern,
                    exclude_bos=args.exclude_bos,
                    exclude_current_token=args.exclude_current_token,
                    error_measure=args.error_measure,
                )
                prev_scores[layer_idx, head_idx] += compute_head_score(
                    head_attn,
                    detection_pattern=prev_pattern,
                    exclude_bos=args.exclude_bos,
                    exclude_current_token=args.exclude_current_token,
                    error_measure=args.error_measure,
                )

    induction_scores /= input_ids.shape[0]
    prev_scores /= input_ids.shape[0]

    induction_flat = [(float(induction_scores[l, h]), l, h) for l in range(num_layers) for h in range(num_heads)]
    prev_flat = [(float(prev_scores[l, h]), l, h) for l in range(num_layers) for h in range(num_heads)]
    induction_flat.sort(key=lambda x: x[0], reverse=True)
    prev_flat.sort(key=lambda x: x[0], reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii") as f:
        f.write("Induction diagnostic (repeated random sequence)\n")
        f.write(f"model={args.model}\n")
        f.write(f"seq_len={args.seq_len} pattern_len={args.pattern_len} batch={args.batch}\n")
        f.write(
            f"vocab_range=[{args.vocab_min},{args.vocab_max}) min_word_length={args.min_word_length} "
            f"leading_space={require_leading_space}\n"
        )
        f.write(
            f"exclude_bos={args.exclude_bos} exclude_current_token={args.exclude_current_token} "
            f"error_measure={args.error_measure}\n"
        )
        f.write("\nTop induction heads (TransformerLens detection):\n")
        for rank, (score, layer, head) in enumerate(induction_flat[: args.top], start=1):
            f.write(f"{rank:02d} L{layer}H{head} score={score:.6f}\n")
        f.write("\nTop previous-token heads (TransformerLens detection):\n")
        for rank, (score, layer, head) in enumerate(prev_flat[: args.top], start=1):
            f.write(f"{rank:02d} L{layer}H{head} score={score:.6f}\n")

    print(f"Wrote report to {args.output}")
    print("\nTop induction heads (TransformerLens detection):")
    for rank, (score, layer, head) in enumerate(induction_flat[: args.top], start=1):
        print(f"{rank:02d} L{layer}H{head} score={score:.6f}")
    print("\nTop previous-token heads (TransformerLens detection):")
    for rank, (score, layer, head) in enumerate(prev_flat[: args.top], start=1):
        print(f"{rank:02d} L{layer}H{head} score={score:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
