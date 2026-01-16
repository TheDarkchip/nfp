#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Certify a single induction head from a model binary.

This script is a small wrapper around
`nfp induction certify_head_model_auto(_nonvacuous)` and optionally
creates a diagnostic prompt model with repeated patterns.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_nfp_cmd(nfp_bin: str | None) -> list[str]:
    if nfp_bin:
        return [nfp_bin]
    env_bin = os.environ.get("NFP_BIN")
    if env_bin:
        return [env_bin]
    local_bin = Path(".lake/build/bin/nfp")
    if local_bin.exists():
        return [str(local_bin)]
    return ["lake", "exe", "nfp"]


def ensure_model(
    model_path: Path,
    *,
    seq_len: int,
    pattern_len: int,
    seed: int,
    vocab_min: int,
    vocab_max: int,
    min_word_length: int,
    allow_no_leading_space: bool,
    model_name: str,
) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    generator = [
        sys.executable,
        "scripts/generate_rigorous_induction.py",
        "--output",
        str(model_path),
        "--seq-len",
        str(seq_len),
        "--pattern-len",
        str(pattern_len),
        "--seed",
        str(seed),
        "--vocab-min",
        str(vocab_min),
        "--vocab-max",
        str(vocab_max),
        "--min-word-length",
        str(min_word_length),
        "--model",
        model_name,
    ]
    if allow_no_leading_space:
        generator.append("--allow-no-leading-space")
    if shutil.which("uv"):
        generator = ["uv", "run"] + generator
    subprocess.run(generator, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Certify an induction head from a model binary."
    )
    parser.add_argument("--model", default="models/gpt2_rigorous.nfpt")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--period", type=int)
    parser.add_argument("--nonvacuous", action="store_true")
    parser.add_argument("--zero-based", action="store_true")
    parser.add_argument("--min-active", type=int)
    parser.add_argument("--min-logit-diff", type=str)
    parser.add_argument("--min-margin", type=str)
    parser.add_argument("--max-eps", type=str)
    parser.add_argument("--nfp-bin", help="Path to nfp binary")

    parser.add_argument(
        "--ensure-model",
        action="store_true",
        help="Generate a diagnostic model if the path does not exist",
    )
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--pattern-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-min", type=int, default=1000)
    parser.add_argument("--vocab-max", type=int, default=5000)
    parser.add_argument("--min-word-length", type=int, default=4)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--allow-no-leading-space", action="store_true")

    args = parser.parse_args()

    model_path = Path(args.model)
    if args.ensure_model:
        ensure_model(
            model_path,
            seq_len=args.seq_len,
            pattern_len=args.pattern_len,
            seed=args.seed,
            vocab_min=args.vocab_min,
            vocab_max=args.vocab_max,
            min_word_length=args.min_word_length,
            allow_no_leading_space=args.allow_no_leading_space,
            model_name=args.model_name,
        )
    if not model_path.exists():
        print(f"error: model not found at {model_path}", file=sys.stderr)
        return 1

    subcmd = "certify_head_model_auto"
    if args.nonvacuous:
        subcmd = "certify_head_model_auto_nonvacuous"

    cmd = resolve_nfp_cmd(args.nfp_bin) + [
        "induction",
        subcmd,
        "--model",
        str(model_path),
        "--layer",
        str(args.layer),
        "--head",
        str(args.head),
    ]
    if args.zero_based:
        cmd.append("--zero-based")
    if args.period is not None:
        cmd += ["--period", str(args.period)]
    if args.min_active is not None:
        cmd += ["--min-active", str(args.min_active)]
    if args.min_logit_diff is not None:
        cmd += ["--min-logit-diff", args.min_logit_diff]
    if args.min_margin is not None:
        cmd += ["--min-margin", args.min_margin]
    if args.max_eps is not None:
        cmd += ["--max-eps", args.max_eps]

    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
