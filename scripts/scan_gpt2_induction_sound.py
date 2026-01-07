#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Scan GPT-2 induction head candidates with SOUND logit-diff bounds.

This script:
1) Ensures a GPT-2 "rigorous induction" binary model exists.
2) Uses the untrusted discovery helper to propose head/direction candidates.
3) Runs `nfp induction certify_head_model_nonvacuous` to check each candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout


def ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    generator = [sys.executable, "scripts/generate_rigorous_induction.py", str(model_path)]
    if shutil.which("uv"):
        generator = ["uv", "run"] + generator
    subprocess.run(generator, check=True)


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


def read_header_and_tokens(path: Path) -> tuple[dict[str, str], list[int]]:
    header: dict[str, str] = {}
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF while reading header")
            text = line.decode("ascii").strip()
            if text == "BINARY_START":
                break
            if "=" in text:
                key, value = text.split("=", 1)
                header[key.strip()] = value.strip()
        seq_len_raw = header.get("seq_len")
        if seq_len_raw is None:
            raise ValueError("header missing seq_len")
        seq_len = int(seq_len_raw)
        token_bytes = f.read(seq_len * 4)
        if len(token_bytes) != seq_len * 4:
            raise ValueError("unexpected EOF while reading tokens")
        tokens = list(struct.unpack("<" + "i" * seq_len, token_bytes))
    return header, tokens


def derive_target_negative(tokens: list[int]) -> tuple[int, int]:
    if len(tokens) < 2:
        raise ValueError("need at least 2 tokens to derive target/negative")
    last = tokens[-1]
    prev_idx = None
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] == last:
            prev_idx = i
            break
    if prev_idx is not None and prev_idx + 1 < len(tokens):
        target = tokens[prev_idx + 1]
    else:
        target = last
    negative = tokens[-2]
    if negative == target:
        negative = last if last != target else (0 if target != 0 else 1)
    return target, negative


def parse_logit_lb(output: str) -> Fraction | None:
    for line in output.splitlines():
        if "logitDiffLB=" not in line:
            continue
        for token in line.split():
            if token.startswith("logitDiffLB="):
                value = token.split("=", 1)[1].strip("),")
                try:
                    return Fraction(value)
                except ValueError:
                    return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gpt2_rigorous.nfpt")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--maxSeqLen", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--nfp-bin", help="Path to nfp binary (defaults to .lake/build/bin/nfp)")
    parser.add_argument("--min-eps", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--min-logit-lb", type=float, default=0.0)
    parser.add_argument("--layers", help="Comma-separated layer list or 'all'")
    parser.add_argument("--heads", help="Comma-separated head list or 'all'")
    parser.add_argument("--period", type=int)
    parser.add_argument("--output", default="reports/gpt2_induction_sound_scan.txt")
    args = parser.parse_args()
    args.jobs = max(1, args.jobs)
    top_arg = any(a.startswith("--top") for a in sys.argv[1:])
    max_seq_len_arg = any(a.startswith("--maxSeqLen") for a in sys.argv[1:])
    if args.fast and not top_arg and args.top == parser.get_default("top"):
        args.top = 4

    model_path = Path(args.model)
    ensure_model(model_path)
    nfp_cmd = resolve_nfp_cmd(args.nfp_bin)

    header, tokens = read_header_and_tokens(model_path)
    seq_len = int(header.get("seq_len", "0"))
    if args.fast and not max_seq_len_arg and args.maxSeqLen == parser.get_default("maxSeqLen"):
        if seq_len <= 128:
            args.maxSeqLen = 128
    if seq_len > args.maxSeqLen:
        print(
            f"Error: seq_len {seq_len} exceeds maxSeqLen {args.maxSeqLen}",
            file=sys.stderr,
        )
        return 1
    target, negative = derive_target_negative(tokens)

    discover_json = Path(args.output).with_suffix(".json")
    discover_txt = Path(args.output).with_suffix(".discover.txt")
    discover_cmd = [
        sys.executable,
        "scripts/discover_gpt2_induction_targets.py",
        "--model",
        str(model_path),
        "--top",
        str(args.top),
        "--min-eps",
        str(args.min_eps),
        "--min-margin",
        str(args.min_margin),
        "--min-logit-lb",
        str(args.min_logit_lb),
        "--output",
        str(discover_txt),
        "--json-out",
        str(discover_json),
    ]
    if args.layers is not None:
        discover_cmd += ["--layers", args.layers]
    if args.heads is not None:
        discover_cmd += ["--heads", args.heads]
    if args.period is not None:
        discover_cmd += ["--period", str(args.period)]
    run_cmd(discover_cmd)
    payload = json.loads(discover_json.read_text(encoding="ascii"))
    candidates = payload.get("results", [])
    if not candidates:
        print("No induction candidates found.", file=sys.stderr)
        return 1

    results: list[tuple[Fraction, dict[str, int]]] = []

    def run_cert(candidate: dict[str, int]) -> tuple[dict[str, int], Fraction | None]:
        layer = int(candidate["layer"])
        head = int(candidate["head"])
        target_id = int(candidate.get("target", target))
        negative_id = int(candidate.get("negative", negative))
        cmd = nfp_cmd + [
            "induction",
            "certify_head_model_nonvacuous",
            "--model",
            str(model_path),
            "--layer",
            str(layer),
            "--head",
            str(head),
            "--direction-target",
            str(target_id),
            "--direction-negative",
            str(negative_id),
        ]
        if args.period is not None:
            cmd += ["--period", str(args.period)]
        try:
            cert_out = run_cmd(cmd)
        except subprocess.CalledProcessError:
            return candidate, None
        return candidate, parse_logit_lb(cert_out)

    if args.jobs == 1:
        for candidate in candidates:
            candidate_out, logit_lb = run_cert(candidate)
            if logit_lb is None:
                continue
            results.append((logit_lb, candidate_out))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_cert, candidate): candidate for candidate in candidates}
            for future in as_completed(futures):
                candidate_out, logit_lb = future.result()
                if logit_lb is None:
                    continue
                results.append((logit_lb, candidate_out))

    if not results:
        print("No sound logit bounds produced.", file=sys.stderr)
        return 1

    results.sort(key=lambda x: x[0], reverse=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="ascii") as f:
        f.write("SOUND induction scan (logitDiffLB ranking)\n")
        f.write(f"model={model_path}\n")
        f.write(f"target={target} negative={negative}\n")
        eps_header = header.get("layer_norm_eps") or header.get("eps") or "unknown"
        f.write(f"top={args.top} eps={eps_header}\n")
        for rank, (lb, candidate) in enumerate(results, start=1):
            layer = int(candidate["layer"])
            head = int(candidate["head"])
            target_id = int(candidate.get("target", target))
            negative_id = int(candidate.get("negative", negative))
            f.write(
                f"{rank:02d} L{layer}H{head} "
                f"target={target_id} negative={negative_id} logitDiffLB={lb}\n"
            )

    print(f"Report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
