#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Scan GPT-2 for induction heads with TransformerLens detection and emit certificates.

Workflow:
- build a batch of repeated token sequences (pattern repeated twice),
- score heads with the TransformerLens induction detection pattern ("mul" or "abs"),
- write a TL score report,
- generate Lean-verifiable certificates for top-K heads across one or more prompts,
- emit a batch file for `nfp induction verify --batch`.

Certificates are produced by calling scripts/build_gpt2_induction_cert.py with --tokens-in.
By default, the script emits induction-aligned certificates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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


def write_tokens(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {len(tokens)}\n")
        for idx, tok in enumerate(tokens.tolist()):
            f.write(f"token {idx} {tok}\n")


def score_heads_from_cache(
    cache,
    input_ids: torch.Tensor,
    *,
    num_layers: int,
    num_heads: int,
    exclude_bos: bool,
    exclude_current_token: bool,
    error_measure: str,
) -> np.ndarray:
    scores = np.zeros((num_layers, num_heads), dtype=np.float64)
    for b in range(input_ids.shape[0]):
        tokens = input_ids[b : b + 1]
        det_pat = head_detector.get_induction_head_detection_pattern(tokens.cpu()).to(input_ids.device)
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


def score_prompt_from_cache(
    cache,
    input_ids: torch.Tensor,
    prompt_index: int,
    *,
    num_layers: int,
    num_heads: int,
    exclude_bos: bool,
    exclude_current_token: bool,
    error_measure: str,
) -> np.ndarray:
    scores = np.zeros((num_layers, num_heads), dtype=np.float64)
    tokens = input_ids[prompt_index : prompt_index + 1]
    det_pat = head_detector.get_induction_head_detection_pattern(tokens.cpu()).to(input_ids.device)
    for layer in range(num_layers):
        layer_attn = cache["pattern", layer, "attn"][prompt_index]
        for head in range(num_heads):
            scores[layer, head] = head_detector.compute_head_attention_similarity_score(
                layer_attn[head],
                detection_pattern=det_pat,
                exclude_bos=exclude_bos,
                exclude_current_token=exclude_current_token,
                error_measure=error_measure,
            )
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--pattern-len", type=int, default=25)
    parser.add_argument("--batch", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-min", type=int, default=1000)
    parser.add_argument("--vocab-max", type=int, default=5000)
    parser.add_argument("--min-word-length", type=int, default=4)
    parser.add_argument("--require-leading-space", action="store_true", default=True)
    parser.add_argument("--allow-no-leading-space", action="store_true")
    parser.add_argument("--exclude-bos", action="store_true")
    parser.add_argument("--exclude-current-token", action="store_true")
    parser.add_argument("--error-measure", choices=("mul", "abs"), default="mul")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--cert-dir", type=Path, default=Path("reports/tl_scan"))
    parser.add_argument("--report", type=Path, default=Path("reports/tl_scan/tl_scores.json"))
    parser.add_argument("--batch-out", type=Path, default=Path("reports/tl_scan/tl_scan.batch"))
    parser.add_argument("--batch-min-stripe-mean", default=None)
    parser.add_argument("--batch-min-copying", default=None)
    parser.add_argument("--batch-min-logit-diff", default=None)
    parser.add_argument("--batch-min-margin", default=None)
    parser.add_argument("--batch-max-eps", default=None)
    parser.add_argument("--batch-min-active", default=None)
    parser.add_argument(
        "--cert-kind",
        choices=["onehot-approx", "induction-aligned"],
        default="induction-aligned",
        help="Certificate kind to emit for top heads (default: induction-aligned).",
    )
    parser.add_argument("--decimals", type=int, default=6)
    parser.add_argument("--active-eps-max", default="1/2")
    parser.add_argument("--min-margin", default="0")
    parser.add_argument("--cert-prompts", type=int, default=1,
                        help="How many prompts from the batch to certify (default: 1).")
    parser.add_argument("--cert-indices", type=int, nargs="*",
                        help="Optional explicit prompt indices to certify.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.seq_len <= 0:
        raise SystemExit("seq-len must be positive")
    if args.pattern_len <= 0:
        raise SystemExit("pattern-len must be positive")
    if args.seq_len != 2 * args.pattern_len:
        raise SystemExit("seq-len must equal 2 * pattern-len")

    require_leading_space = args.require_leading_space and not args.allow_no_leading_space
    rng = np.random.default_rng(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
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

    if args.cert_prompts <= 0:
        raise SystemExit("cert-prompts must be positive")
    if args.cert_indices is not None and len(args.cert_indices) == 0:
        raise SystemExit("cert-indices must be non-empty if provided")

    input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=args.device)

    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids, remove_batch_dim=False)

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    scores = score_heads_from_cache(
        cache,
        input_ids,
        num_layers=num_layers,
        num_heads=num_heads,
        exclude_bos=args.exclude_bos,
        exclude_current_token=args.exclude_current_token,
        error_measure=args.error_measure,
    )

    flat = [(float(scores[l, h]), l, h) for l in range(scores.shape[0]) for h in range(scores.shape[1])]
    flat.sort(key=lambda x: x[0], reverse=True)

    if args.cert_indices is None:
        prompt_indices = list(range(min(args.cert_prompts, args.batch)))
    else:
        prompt_indices = args.cert_indices
    for idx in prompt_indices:
        if idx < 0 or idx >= args.batch:
            raise SystemExit("cert-indices out of range for batch")

    tokens_dir = args.cert_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    prompt_tokens = {}
    for idx in prompt_indices:
        tokens_path = tokens_dir / f"prompt_{idx}.tokens"
        prompt_tokens[idx] = tokens_path
        if not args.dry_run:
            write_tokens(tokens_path, batch_tokens[idx])

    prompt_scores = {}
    for idx in prompt_indices:
        scores_prompt = score_prompt_from_cache(
            cache,
            input_ids,
            idx,
            num_layers=num_layers,
            num_heads=num_heads,
            exclude_bos=args.exclude_bos,
            exclude_current_token=args.exclude_current_token,
            error_measure=args.error_measure,
        )
        prompt_scores[idx] = scores_prompt

    report_entries = []
    for rank, (score, layer, head) in enumerate(flat[: args.top], start=1):
        per_prompt = {}
        for idx in prompt_indices:
            per_prompt[str(idx)] = float(prompt_scores[idx][layer, head])
        report_entries.append(
            {
                "rank": rank,
                "layer": layer,
                "head": head,
                "score": score,
                "scores_by_prompt": per_prompt,
            }
        )

    args.cert_dir.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "model": args.model,
        "seq_len": args.seq_len,
        "pattern_len": args.pattern_len,
        "batch": args.batch,
        "seed": args.seed,
        "vocab_range": [args.vocab_min, args.vocab_max],
        "min_word_length": args.min_word_length,
        "leading_space": require_leading_space,
        "exclude_bos": args.exclude_bos,
        "exclude_current_token": args.exclude_current_token,
        "error_measure": args.error_measure,
        "cert_indices": prompt_indices,
        "cert_tokens": {str(k): str(v) for k, v in prompt_tokens.items()},
        "batch_out": str(args.batch_out),
        "top": report_entries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_payload, indent=2), encoding="ascii")
    print(f"Wrote TL score report to {args.report}")

    if args.dry_run:
        return 0

    batch_items = []
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    for entry in report_entries:
        layer = entry["layer"]
        head = entry["head"]
        certs_for_head = {}
        for idx in prompt_indices:
            cert_path = args.cert_dir / "certs" / f"L{layer}H{head}" / f"prompt_{idx}.cert"
            cert_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "scripts/build_gpt2_induction_cert.py",
                "--output",
                str(cert_path),
                "--layer",
                str(layer),
                "--head",
                str(head),
                "--seq",
                str(args.seq_len),
                "--pattern-length",
                str(args.pattern_len),
                "--tokens-in",
                str(prompt_tokens[idx]),
                "--kind",
                args.cert_kind,
                "--decimals",
                str(args.decimals),
                "--model",
                args.model,
                "--device",
                args.device,
                "--active-eps-max",
                str(args.active_eps_max),
                "--min-margin",
                str(args.min_margin),
            ]
            print(f"Building cert for L{layer}H{head} (prompt {idx})...")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                print(result.stderr.strip())
                print(result.stdout.strip())
                print(f"warning: certificate failed for L{layer}H{head} prompt {idx}")
                continue
            certs_for_head[str(idx)] = str(cert_path)
            batch_items.append((cert_path, prompt_tokens[idx]))
        entry["cert_paths"] = certs_for_head

    args.report.write_text(json.dumps(report_payload, indent=2), encoding="ascii")

    args.batch_out.parent.mkdir(parents=True, exist_ok=True)
    with args.batch_out.open("w", encoding="ascii") as f:
        if args.batch_min_active is not None:
            f.write(f"min-active {args.batch_min_active}\n")
        if args.batch_min_logit_diff is not None:
            f.write(f"min-logit-diff {args.batch_min_logit_diff}\n")
        if args.batch_min_margin is not None:
            f.write(f"min-margin {args.batch_min_margin}\n")
        if args.batch_max_eps is not None:
            f.write(f"max-eps {args.batch_max_eps}\n")
        if args.batch_min_stripe_mean is not None:
            f.write(f"min-stripe-mean {args.batch_min_stripe_mean}\n")
        if args.batch_min_copying is not None:
            f.write(f"min-copying {args.batch_min_copying}\n")
        for cert_path, tokens_path in batch_items:
            f.write(f"item {cert_path} {tokens_path}\n")
    print(f"Wrote batch file to {args.batch_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
