#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build a batch of induction-head certificates for GPT-2 small.

This script shells out to build_gpt2_induction_cert.py for each seed and then
writes a batch file listing the resulting certs and token lists. Failed seeds
are skipped until the requested count is reached (bounded by --max-attempts).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-out", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Max seeds to try (default: count * 5).")
    parser.add_argument("--layer", type=int, required=True, help="0-based layer index")
    parser.add_argument("--head", type=int, required=True, help="0-based head index")
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--pattern-length", type=int, default=16)
    parser.add_argument("--random-pattern", action="store_true")
    parser.add_argument("--active-eps-max", default="1/2")
    parser.add_argument("--min-margin", default="0")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kind", default="onehot-approx",
                        choices=["onehot-approx", "induction-aligned"],
                        help="Certificate kind (default: onehot-approx).")
    parser.add_argument("--search-direction", action="store_true")
    parser.add_argument("--direction-vocab-min", type=int, default=1000)
    parser.add_argument("--direction-vocab-max", type=int, default=2000)
    parser.add_argument("--direction-min-lb", default="0")
    parser.add_argument("--direction-topk", type=int, default=10)
    parser.add_argument("--min-logit-diff", default=None)
    parser.add_argument("--min-avg-logit-diff", default=None)
    parser.add_argument("--min-active", type=int, default=None)
    parser.add_argument("--max-eps", default=None)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_out = args.batch_out or (out_dir / "gpt2_induction.batch")

    items: list[tuple[Path, Path]] = []
    script = Path(__file__).with_name("build_gpt2_induction_cert.py")

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = args.count * 5

    successes = 0
    attempt = 0
    seed = args.seed_start
    while successes < args.count and attempt < max_attempts:
        cert_path = out_dir / f"cert_{successes:04d}.cert"
        tokens_path = out_dir / f"tokens_{successes:04d}.tokens"
        cmd = [
            "python",
            str(script),
            "--output",
            str(cert_path),
            "--layer",
            str(args.layer),
            "--head",
            str(args.head),
            "--seq",
            str(args.seq),
            "--pattern-length",
            str(args.pattern_length),
            "--seed",
            str(seed),
            "--active-eps-max",
            str(args.active_eps_max),
            "--min-margin",
            str(args.min_margin),
            "--tokens-out",
            str(tokens_path),
            "--kind",
            str(args.kind),
            "--model",
            str(args.model),
            "--device",
            str(args.device),
        ]
        if args.random_pattern:
            cmd.append("--random-pattern")
        if args.search_direction:
            cmd.extend(
                [
                    "--search-direction",
                    "--direction-vocab-min",
                    str(args.direction_vocab_min),
                    "--direction-vocab-max",
                    str(args.direction_vocab_max),
                    "--direction-min-lb",
                    str(args.direction_min_lb),
                    "--direction-topk",
                    str(args.direction_topk),
                ]
            )
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            items.append((cert_path, tokens_path))
            successes += 1
        else:
            print(f"warning: certificate generation failed for seed {seed}")
        attempt += 1
        seed += 1

    if successes < args.count:
        raise SystemExit(
            f"only generated {successes} certs after {attempt} attempts"
        )

    lines = []
    if args.min_active is not None:
        lines.append(f"min-active {args.min_active}")
    if args.min_logit_diff is not None:
        lines.append(f"min-logit-diff {args.min_logit_diff}")
    if args.min_avg_logit_diff is not None:
        lines.append(f"min-avg-logit-diff {args.min_avg_logit_diff}")
    if args.max_eps is not None:
        lines.append(f"max-eps {args.max_eps}")
    for cert_path, tokens_path in items:
        lines.append(f"item {cert_path} {tokens_path}")

    batch_out.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote batch file to {batch_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
