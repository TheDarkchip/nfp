#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Score circuit-level candidate pairs (prev-token head + induction head) using
TransformerLens head detection patterns.

This computes mean TL scores for previous-token heads and induction heads
across multiple random repeated-pattern prompts, then ranks candidate pairs
with prev_layer < induction_layer. Optionally writes summary JSON.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import GPT2Tokenizer
    from transformer_lens import HookedTransformer
    import transformer_lens.head_detector as head_detector
except ImportError as exc:
    raise SystemExit(
        "Requires transformer-lens + torch + transformers (use `uv add transformer-lens torch transformers`)."
    ) from exc


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


def score_heads(
    cache,
    input_ids: torch.Tensor,
    *,
    num_layers: int,
    num_heads: int,
    detection_pattern_fn,
    exclude_bos: bool,
    exclude_current_token: bool,
    error_measure: str,
) -> np.ndarray:
    scores = np.zeros((num_layers, num_heads), dtype=np.float64)
    for b in range(input_ids.shape[0]):
        tokens = input_ids[b : b + 1]
        det_pat = detection_pattern_fn(tokens.cpu()).to(input_ids.device)
        for layer in range(num_layers):
            layer_attn = cache["pattern", layer, "attn"][b]
            for head in range(num_heads):
                score = head_detector.compute_head_attention_similarity_score(
                    layer_attn[head],
                    detection_pattern=det_pat,
                    exclude_bos=exclude_bos,
                    exclude_current_token=exclude_current_token,
                    error_measure=error_measure,
                )
                scores[layer, head] += score
    scores /= input_ids.shape[0]
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--pattern-len", type=int, default=16)
    parser.add_argument("--batch", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024, 42])
    parser.add_argument("--vocab-min", type=int, default=1000)
    parser.add_argument("--vocab-max", type=int, default=5000)
    parser.add_argument("--min-word-length", type=int, default=4)
    parser.add_argument("--require-leading-space", action="store_true", default=True)
    parser.add_argument("--allow-no-leading-space", action="store_true")
    parser.add_argument("--exclude-bos", action="store_true")
    parser.add_argument("--exclude-current-token", action="store_true")
    parser.add_argument("--error-measure", choices=("mul", "abs"), default="mul")
    parser.add_argument("--top-heads", type=int, default=20)
    parser.add_argument("--top-pairs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("reports/tl_scan/circuit_pairs.json"))
    args = parser.parse_args()

    require_leading_space = args.require_leading_space and not args.allow_no_leading_space
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    model.eval()

    candidates = select_vocab_candidates(
        tokenizer,
        vocab_min=args.vocab_min,
        vocab_max=args.vocab_max,
        min_word_length=args.min_word_length,
        require_leading_space=require_leading_space,
    )

    prev_scores_by_head = defaultdict(list)
    ind_scores_by_head = defaultdict(list)

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        batch_tokens = build_batch(
            rng,
            candidates,
            batch_size=args.batch,
            pattern_len=args.pattern_len,
            seq_len=args.seq_len,
        )
        input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=args.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, remove_batch_dim=False)
        num_layers = model.cfg.n_layers
        num_heads = model.cfg.n_heads
        prev_scores = score_heads(
            cache,
            input_ids,
            num_layers=num_layers,
            num_heads=num_heads,
            detection_pattern_fn=head_detector.get_previous_token_head_detection_pattern,
            exclude_bos=args.exclude_bos,
            exclude_current_token=args.exclude_current_token,
            error_measure=args.error_measure,
        )
        ind_scores = score_heads(
            cache,
            input_ids,
            num_layers=num_layers,
            num_heads=num_heads,
            detection_pattern_fn=head_detector.get_induction_head_detection_pattern,
            exclude_bos=args.exclude_bos,
            exclude_current_token=args.exclude_current_token,
            error_measure=args.error_measure,
        )
        for layer in range(num_layers):
            for head in range(num_heads):
                prev_scores_by_head[(layer, head)].append(float(prev_scores[layer, head]))
                ind_scores_by_head[(layer, head)].append(float(ind_scores[layer, head]))

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    prev_means = {k: mean(v) for k, v in prev_scores_by_head.items()}
    ind_means = {k: mean(v) for k, v in ind_scores_by_head.items()}

    prev_heads = sorted(
        [(score, layer, head) for (layer, head), score in prev_means.items()],
        reverse=True,
    )[: args.top_heads]
    ind_heads = sorted(
        [(score, layer, head) for (layer, head), score in ind_means.items()],
        reverse=True,
    )[: args.top_heads]

    pairs = []
    for prev_score, prev_layer, prev_head in prev_heads:
        for ind_score, ind_layer, ind_head in ind_heads:
            if prev_layer >= ind_layer:
                continue
            pair_score = prev_score * ind_score
            pairs.append(
                {
                    "pair_score": pair_score,
                    "prev": {"layer": prev_layer, "head": prev_head, "score": prev_score},
                    "induction": {"layer": ind_layer, "head": ind_head, "score": ind_score},
                }
            )
    pairs.sort(key=lambda x: x["pair_score"], reverse=True)

    payload = {
        "model": args.model,
        "seq_len": args.seq_len,
        "pattern_len": args.pattern_len,
        "batch": args.batch,
        "seeds": args.seeds,
        "vocab_range": [args.vocab_min, args.vocab_max],
        "min_word_length": args.min_word_length,
        "leading_space": require_leading_space,
        "exclude_bos": args.exclude_bos,
        "exclude_current_token": args.exclude_current_token,
        "error_measure": args.error_measure,
        "top_prev_heads": [
            {"rank": i + 1, "layer": layer, "head": head, "score": score}
            for i, (score, layer, head) in enumerate(prev_heads)
        ],
        "top_induction_heads": [
            {"rank": i + 1, "layer": layer, "head": head, "score": score}
            for i, (score, layer, head) in enumerate(ind_heads)
        ],
        "top_pairs": pairs[: args.top_pairs],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="ascii")
    print(f"Wrote {args.output}")
    print("Top circuit pairs:")
    for entry in pairs[: args.top_pairs]:
        prev = entry["prev"]
        ind = entry["induction"]
        print(
            f"prev L{prev['layer']}H{prev['head']} ({prev['score']:.4f}) + "
            f"ind L{ind['layer']}H{ind['head']} ({ind['score']:.4f}) -> "
            f"score={entry['pair_score']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
