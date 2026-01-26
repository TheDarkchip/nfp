#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Score circuit-level candidate pairs (prev-token head + induction head) using
TransformerLens head detection patterns.

This computes mean TL scores for previous-token heads and induction heads
across multiple random repeated-pattern prompts, then ranks candidate pairs
with prev_layer < induction_layer. Optionally writes summary JSON.

If requested, emits a shell script that builds certs, optionally verifies them
via `nfp induction verify`, and runs `nfp induction verify-circuit` using
provided token files.
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


def write_tokens(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {len(tokens)}\n")
        for idx, tok in enumerate(tokens):
            f.write(f"token {idx} {tok}\n")


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
    parser.add_argument("--emit-verify-script", action="store_true",
                        help="Emit a shell script that builds certs and runs verify-circuit.")
    parser.add_argument("--verify-script-out", type=Path,
                        default=Path("reports/tl_scan/circuit_pairs_verify.sh"))
    parser.add_argument("--cert-dir", type=Path, default=Path("reports/tl_scan/circuit_certs"))
    parser.add_argument("--top-verify", type=int, default=None,
                        help="How many top pairs to include in the verify script (default: top-pairs).")
    parser.add_argument("--period", type=int, default=None,
                        help="Period to use for circuit verification (default: pattern-len).")
    parser.add_argument("--prev-tokens", type=Path, default=Path("reports/prev.tokens"))
    parser.add_argument("--induction-tokens", type=Path,
                        default=Path("reports/induction32.tokens"))
    parser.add_argument("--prev-const-token", type=int, default=1,
                        help="Token id to use for constant prev-token prompts.")
    parser.add_argument("--write-tokens", action="store_true",
                        help="Write default prev/induction token files if missing.")
    parser.add_argument("--cert-active-eps-max", default="1")
    parser.add_argument("--cert-min-margin", default="-10000")
    parser.add_argument("--emit-model-slice", action="store_true",
                        help="Include model slice data in emitted certs.")
    parser.add_argument("--model-decimals", type=int, default=None,
                        help="Decimal rounding for model slice entries.")
    parser.add_argument("--model-ln-slack", default=None,
                        help="Slack for LayerNorm bounds when emitting model slice.")
    parser.add_argument("--model-ln-scale", type=int, default=None,
                        help="Optional LayerNorm sqrt bound scale (positive).")
    parser.add_argument("--model-ln-fast", action="store_true",
                        help="Use the fixed-denominator fast path in the verifier.")
    parser.add_argument("--verify-head-certs", action="store_true",
                        help="Run `nfp induction verify` on each head cert before circuit checks.")
    parser.add_argument("--verify-min-logit-diff", default=None,
                        help=("Minimum logit-diff lower bound to require when verifying "
                              "induction head certs (optional)."))
    parser.add_argument("--direction-target", type=int, default=None,
                        help="Token id for logit-diff direction target (optional).")
    parser.add_argument("--direction-negative", type=int, default=None,
                        help="Token id for logit-diff direction negative (optional).")
    parser.add_argument("--search-direction", action="store_true",
                        help="Search for a logit-diff direction within a vocab range.")
    parser.add_argument("--direction-vocab-min", type=int, default=1000,
                        help="Minimum vocab id for direction search (inclusive).")
    parser.add_argument("--direction-vocab-max", type=int, default=2000,
                        help="Maximum vocab id for direction search (exclusive).")
    parser.add_argument("--direction-max-candidates", type=int, default=0,
                        help="Limit number of direction candidates (0 = all in range).")
    parser.add_argument("--direction-min-lb", default=None,
                        help=("Minimum logit-diff lower bound to accept "
                              "(omit to accept any LB)."))
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

    if args.emit_verify_script:
        if (args.direction_target is None) != (args.direction_negative is None):
            raise SystemExit("direction-target and direction-negative must be provided together")
        if args.search_direction and args.direction_target is not None:
            raise SystemExit("choose either search-direction or explicit direction-target/negative")
        if args.emit_model_slice and not (args.search_direction or args.direction_target is not None):
            raise SystemExit(
                "emit-model-slice verify script requires --search-direction or --direction-target/negative"
            )
        if args.verify_min_logit_diff is not None and not (
            args.search_direction or args.direction_target is not None
        ):
            raise SystemExit("verify-min-logit-diff requires direction metadata")
        period = args.pattern_len if args.period is None else args.period
        top_verify = args.top_pairs if args.top_verify is None else args.top_verify
        if args.seq_len != 2 * period:
            raise SystemExit("seq-len must equal 2 * period for circuit verification")
        if args.write_tokens:
            if args.vocab_min + period > args.vocab_max:
                raise SystemExit("vocab range too small to write default induction tokens")
            induction_tokens = list(range(args.vocab_min, args.vocab_min + period)) * 2
            prev_tokens = [args.prev_const_token] * args.seq_len
            write_tokens(args.induction_tokens, induction_tokens)
            write_tokens(args.prev_tokens, prev_tokens)
        else:
            if not args.induction_tokens.exists():
                raise SystemExit(f"missing induction tokens file: {args.induction_tokens}")
            if not args.prev_tokens.exists():
                raise SystemExit(f"missing prev tokens file: {args.prev_tokens}")
        verify_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        prev_cert_dir = args.cert_dir / "prev"
        ind_cert_dir = args.cert_dir / "ind"
        prev_cert_dir.mkdir(parents=True, exist_ok=True)
        ind_cert_dir.mkdir(parents=True, exist_ok=True)
        verify_lines.append("# Build prev-token certs once")
        model_args = []
        direction_args = []
        if args.emit_model_slice:
            model_args.append("--emit-model-slice")
            if args.model_decimals is not None:
                model_args.extend(["--model-decimals", str(args.model_decimals)])
            if args.model_ln_slack is not None:
                model_args.extend(["--model-ln-slack", str(args.model_ln_slack)])
            if args.model_ln_scale is not None:
                model_args.extend(["--model-ln-scale", str(args.model_ln_scale)])
            if args.model_ln_fast:
                model_args.append("--model-ln-fast")
        if args.search_direction:
            direction_args.append("--search-direction")
            direction_args.extend(["--direction-vocab-min", str(args.direction_vocab_min)])
            direction_args.extend(["--direction-vocab-max", str(args.direction_vocab_max)])
            direction_args.extend(["--direction-max-candidates", str(args.direction_max_candidates)])
            if args.direction_min_lb is not None:
                direction_args.extend(["--direction-min-lb", str(args.direction_min_lb)])
        elif args.direction_target is not None:
            direction_args.extend(["--direction-target", str(args.direction_target)])
            direction_args.extend(["--direction-negative", str(args.direction_negative)])
        if (args.search_direction or args.direction_target is not None) and not args.emit_model_slice:
            direction_args.append("--omit-unembed-rows")
        seen_prev = set()
        for entry in pairs[:top_verify]:
            prev = entry["prev"]
            key = (prev["layer"], prev["head"])
            if key in seen_prev:
                continue
            seen_prev.add(key)
            verify_lines.append(
                f"python scripts/build_gpt2_induction_cert.py --output {prev_cert_dir}/L{key[0]}H{key[1]}.cert "
                f"--layer {key[0]} --head {key[1]} --seq {args.seq_len} "
                f"--pattern-length {period} --tokens-in {args.prev_tokens} "
                f"--active-eps-max {args.cert_active_eps_max} --min-margin {args.cert_min_margin} "
                + " ".join(model_args + direction_args)
            )
        verify_lines.append("")
        verify_lines.append("# Build induction certs and verify circuit pairs")
        verify_args = [
            "--min-margin", str(args.cert_min_margin),
            "--max-eps", str(args.cert_active_eps_max),
        ]
        verify_args_ind = verify_args[:]
        if args.verify_min_logit_diff is not None:
            verify_args_ind.extend(["--min-logit-diff", str(args.verify_min_logit_diff)])
        for entry in pairs[:top_verify]:
            prev = entry["prev"]
            ind = entry["induction"]
            ind_cert = f"{ind_cert_dir}/L{ind['layer']}H{ind['head']}.cert"
            prev_cert = f"{prev_cert_dir}/L{prev['layer']}H{prev['head']}.cert"
            verify_lines.append(
                f"python scripts/build_gpt2_induction_cert.py --output {ind_cert} "
                f"--layer {ind['layer']} --head {ind['head']} --seq {args.seq_len} "
                f"--pattern-length {period} --tokens-in {args.induction_tokens} --prev-shift "
                f"--active-eps-max {args.cert_active_eps_max} --min-margin {args.cert_min_margin} "
                + " ".join(model_args + direction_args)
            )
            if args.verify_head_certs:
                verify_lines.append(
                    f"lake exe nfp induction verify --cert {prev_cert} "
                    + " ".join(verify_args)
                )
                verify_lines.append(
                    f"lake exe nfp induction verify --cert {ind_cert} "
                    + " ".join(verify_args_ind)
                )
            verify_lines.append(
                f"lake exe nfp induction verify-circuit --prev-cert {prev_cert} "
                f"--ind-cert {ind_cert} --period {period} --tokens {args.induction_tokens}"
            )
        args.verify_script_out.parent.mkdir(parents=True, exist_ok=True)
        args.verify_script_out.write_text("\n".join(verify_lines) + "\n", encoding="ascii")
        print(f"Wrote {args.verify_script_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
