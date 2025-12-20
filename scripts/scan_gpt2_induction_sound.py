#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Scan GPT-2 induction head candidates with SOUND logit-diff bounds.

This script:
1) Ensures a GPT-2 "rigorous induction" binary model exists.
2) Uses the heuristic `induction` command to list candidate head pairs.
3) Runs `induction_cert` for each pair and ranks by logitDiffLB.
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


PAIR_RE = re.compile(r"L(\d+)H(\d+)\s+->\s+L(\d+)H(\d+)")


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout


def ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    generator = ["python", "scripts/generate_rigorous_induction.py", str(model_path)]
    if shutil.which("uv"):
        generator = ["uv", "run"] + generator
    subprocess.run(generator, check=True)


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


def parse_candidates(output: str, top: int) -> list[tuple[int, int, int, int]]:
    pairs: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for line in output.splitlines():
        match = PAIR_RE.search(line)
        if match is None:
            continue
        pair = tuple(int(x) for x in match.groups())
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
        if len(pairs) >= top:
            break
    return pairs


def parse_logit_lb(output: str) -> Fraction | None:
    for line in output.splitlines():
        if line.startswith("logitDiffLB="):
            token = line.split("=", 1)[1].split()[0]
            try:
                return Fraction(token)
            except ValueError:
                return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gpt2_rigorous.nfpt")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--delta", default="0.01")
    parser.add_argument("--coord", type=int, default=0)
    parser.add_argument("--offset1", type=int, default=-1)
    parser.add_argument("--offset2", type=int, default=-1)
    parser.add_argument("--maxSeqLen", type=int, default=256)
    parser.add_argument("--tightPattern", action="store_true")
    parser.add_argument("--tightPatternLayers", type=int)
    parser.add_argument("--perRowPatternLayers", type=int)
    parser.add_argument("--bestMatch", action="store_true")
    parser.add_argument("--queryPos", type=int)
    parser.add_argument("--output", default="reports/gpt2_induction_sound_scan.txt")
    args = parser.parse_args()

    model_path = Path(args.model)
    ensure_model(model_path)

    header, tokens = read_header_and_tokens(model_path)
    seq_len = int(header.get("seq_len", "0"))
    if seq_len > args.maxSeqLen:
        print(
            f"Error: seq_len {seq_len} exceeds maxSeqLen {args.maxSeqLen}",
            file=sys.stderr,
        )
        return 1
    target, negative = derive_target_negative(tokens)

    induction_out = run_cmd(
        [
            "lake",
            "exe",
            "nfp",
            "induction",
            str(model_path),
            "--threshold",
            "0.0",
        ]
    )
    pairs = parse_candidates(induction_out, args.top)
    if not pairs:
        print("No induction candidates found.", file=sys.stderr)
        return 1

    results: list[tuple[Fraction, tuple[int, int, int, int]]] = []
    for (l1, h1, l2, h2) in pairs:
        cmd = [
            "lake",
            "exe",
            "nfp",
            "induction_cert",
            str(model_path),
            "--layer1",
            str(l1),
            "--head1",
            str(h1),
            "--layer2",
            str(l2),
            "--head2",
            str(h2),
            "--coord",
            str(args.coord),
            "--offset1",
            str(args.offset1),
            "--offset2",
            str(args.offset2),
            "--delta",
            args.delta,
            "--maxSeqLen",
            str(args.maxSeqLen),
            "--target",
            str(target),
            "--negative",
            str(negative),
        ]
        if args.tightPattern:
            cmd.append("--tightPattern")
        if args.tightPatternLayers is not None:
            cmd.extend(["--tightPatternLayers", str(args.tightPatternLayers)])
        if args.perRowPatternLayers is not None:
            cmd.extend(["--perRowPatternLayers", str(args.perRowPatternLayers)])
        if args.bestMatch:
            cmd.append("--bestMatch")
        if args.queryPos is not None:
            cmd.extend(["--queryPos", str(args.queryPos)])
        cert_out = run_cmd(cmd)
        logit_lb = parse_logit_lb(cert_out)
        if logit_lb is None:
            continue
        results.append((logit_lb, (l1, h1, l2, h2)))

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
        f.write(
            f"bestMatch={args.bestMatch} queryPos={args.queryPos} "
            f"tightPatternLayers={args.tightPatternLayers} "
            f"perRowPatternLayers={args.perRowPatternLayers}\n"
        )
        eps_header = header.get("layer_norm_eps") or header.get("eps") or "unknown"
        f.write(f"top={args.top} delta={args.delta} eps={eps_header}\n")
        for rank, (lb, (l1, h1, l2, h2)) in enumerate(results, start=1):
            f.write(
                f"{rank:02d} L{l1}H{h1} -> L{l2}H{h2} logitDiffLB={lb}\n"
            )

    print(f"Report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
