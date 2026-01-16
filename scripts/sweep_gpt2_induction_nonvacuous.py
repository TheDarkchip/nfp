#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Sweep prompt parameters and evaluate induction head scores for GPT-2.

This is untrusted orchestration: discovery uses floating-point math and only
Lean verification results are treated as definitive in logit mode.

Layer/head indices are one-based (literature-aligned). `prev` defaults to bigram
prefix matching.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


LOGIT_RE = re.compile(r"logitDiffLB=([^\s\)]+)")


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    logit_lb: str | None
    stdout: str
    stderr: str


def parse_int_list(raw: str, name: str) -> List[int]:
    items: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(int(part))
        except ValueError as exc:
            raise ValueError(f"invalid {name} entry: {part}") from exc
    if not items:
        raise ValueError(f"{name} list is empty")
    return items


def resolve_python_cmd() -> List[str]:
    if shutil.which("uv"):
        return ["uv", "run"]
    return ["python3"]


def resolve_nfp_cmd(nfp_bin: str | None) -> List[str]:
    if nfp_bin:
        return [nfp_bin]
    local_bin = Path(".lake/build/bin/nfp")
    if local_bin.exists():
        return [str(local_bin)]
    return ["lake", "exe", "nfp"]


def run_cmd(cmd: Iterable[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), check=check, capture_output=True, text=True)


def ensure_model(
    generator: Path,
    output: Path,
    seq_len: int,
    pattern_len: int,
    seed: int,
) -> None:
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = resolve_python_cmd() + [
        str(generator),
        "--output",
        str(output),
        "--seq-len",
        str(seq_len),
        "--pattern-len",
        str(pattern_len),
        "--seed",
        str(seed),
    ]
    run_cmd(cmd, check=True)


def run_discovery(
    discover_script: Path,
    model: Path,
    max_tokens: int,
    top: int,
    min_eps: float,
    min_margin: float,
    min_logit_lb: float,
    min_score: float,
    min_copy: float | None,
    score_mode: str,
    period: int | None,
    output_dir: Path,
    prev_mode: str,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_out = output_dir / f"{model.stem}.json"
    cmd = resolve_python_cmd() + [
        str(discover_script),
        "--model",
        str(model),
        "--max-tokens",
        str(max_tokens),
        "--top",
        str(top),
        "--min-eps",
        str(min_eps),
        "--min-margin",
        str(min_margin),
        "--min-logit-lb",
        str(min_logit_lb),
        "--min-score",
        str(min_score),
        "--score-mode",
        score_mode,
        "--json-out",
        str(json_out),
    ]
    if min_copy is not None:
        cmd += ["--min-copy", str(min_copy)]
    if period is not None:
        cmd += ["--period", str(period)]
    if prev_mode != "bigram":
        cmd += ["--prev-mode", prev_mode]
    run_cmd(cmd, check=True)
    payload = json.loads(json_out.read_text(encoding="ascii"))
    return payload.get("results", [])


def verify_candidate(
    nfp_cmd: List[str],
    model: Path,
    layer: int,
    head: int,
    target: int,
    negative: int,
    period: int | None,
) -> VerifyResult:
    cmd = nfp_cmd + [
        "induction",
        "certify_head_model_nonvacuous",
        "--model",
        str(model),
        "--layer",
        str(layer),
        "--head",
        str(head),
        "--direction-target",
        str(target),
        "--direction-negative",
        str(negative),
    ]
    if period is not None:
        cmd += ["--period", str(period)]
    proc = run_cmd(cmd, check=False)
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    logit_lb = None
    match = LOGIT_RE.search(stdout)
    if match:
        logit_lb = match.group(1)
    return VerifyResult(proc.returncode == 0, logit_lb, stdout, stderr)


def write_csv_row(path: Path, row: dict) -> None:
    header = [
        "model_path",
        "seq_len",
        "pattern_len",
        "seed",
        "layer",
        "head",
        "target",
        "negative",
        "score_mode",
        "score",
        "prev_mean",
        "prev_median",
        "prev_top1_frac",
        "copy_mean",
        "copy_weighted_mean",
        "approx_logit_lb",
        "approx_eps",
        "approx_margin",
        "approx_min_prev",
        "approx_value_range",
        "active",
        "period",
        "verify_status",
        "verify_logit_lb",
    ]
    new_file = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="ascii") as f:
        if new_file:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row.get(col, "")) for col in header) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/gpt2_induction_sweep.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--generator", type=Path, default=Path("scripts/generate_rigorous_induction.py"))
    parser.add_argument("--discover", type=Path, default=Path("scripts/discover_gpt2_induction_targets.py"))
    parser.add_argument("--seq-lens", default="64")
    parser.add_argument("--pattern-lens", default="16")
    parser.add_argument("--seeds", default="1337")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--verify-top", type=int, default=3)
    parser.add_argument("--min-eps", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--min-logit-lb", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-copy", type=float)
    parser.add_argument(
        "--score-mode",
        choices=["attn", "copy", "attn_copy", "logit"],
        default="attn",
        help="Rank by attention/copy score or logit-diff bound.",
    )
    parser.add_argument("--use-period", action="store_true",
                        help="Use pattern length as the period override")
    parser.add_argument(
        "--prev-mode",
        choices=["bigram", "token", "period"],
        default="bigram",
        help="Choose prev/active construction (forwarded to discovery).",
    )
    parser.add_argument("--nfp-bin", help="Path to nfp binary")
    parser.add_argument("--discovery-dir", type=Path, default=Path("reports/discovery"))
    args = parser.parse_args()

    seq_lens = parse_int_list(args.seq_lens, "seq-lens")
    pattern_lens = parse_int_list(args.pattern_lens, "pattern-lens")
    seeds = parse_int_list(args.seeds, "seeds")

    nfp_cmd = resolve_nfp_cmd(args.nfp_bin)

    for seq_len in seq_lens:
        for pattern_len in pattern_lens:
            for seed in seeds:
                model_name = f"gpt2_rigorous_seq{seq_len}_pat{pattern_len}_seed{seed}.nfpt"
                model_path = args.model_dir / model_name
                ensure_model(args.generator, model_path, seq_len, pattern_len, seed)
                period = pattern_len if args.use_period else None
                results = run_discovery(
                    args.discover,
                    model_path,
                    args.max_tokens,
                    args.top,
                    args.min_eps,
                    args.min_margin,
                    args.min_logit_lb,
                    args.min_score,
                    args.min_copy,
                    args.score_mode,
                    period,
                    args.discovery_dir,
                    args.prev_mode,
                )
                if not results:
                    print(
                        f"no candidates for seq={seq_len} pat={pattern_len} seed={seed}",
                        flush=True,
                    )
                    continue
                if args.score_mode == "logit":
                    for result in results[: args.verify_top]:
                        layer = result["layer"] - 1
                        head = result["head"] - 1
                        verify = verify_candidate(
                            nfp_cmd,
                            model_path,
                            layer,
                            head,
                            result["target"],
                            result["negative"],
                            period,
                        )
                        status = "ok" if verify.ok else "fail"
                        if verify.ok:
                            print(
                                f"verified L{result['layer']}H{result['head']} "
                                f"seq={seq_len} pat={pattern_len} seed={seed}",
                                flush=True,
                            )
                        row = {
                            "model_path": model_path,
                            "seq_len": seq_len,
                            "pattern_len": pattern_len,
                            "seed": seed,
                            "layer": result["layer"],
                            "head": result["head"],
                            "target": result["target"],
                            "negative": result["negative"],
                            "score_mode": args.score_mode,
                            "score": "",
                            "prev_mean": result.get("prev_mean", ""),
                            "prev_median": result.get("prev_median", ""),
                            "prev_top1_frac": result.get("prev_top1_frac", ""),
                            "copy_mean": result.get("copy_mean", ""),
                            "copy_weighted_mean": result.get("copy_weighted_mean", ""),
                            "approx_logit_lb": result["logit_lb"],
                            "approx_eps": result["eps"],
                            "approx_margin": result["margin"],
                            "approx_min_prev": result["min_prev"],
                            "approx_value_range": result["value_range"],
                            "active": result["active"],
                            "period": period if period is not None else "",
                            "verify_status": status,
                            "verify_logit_lb": verify.logit_lb or "",
                        }
                        write_csv_row(args.output, row)
                        if not verify.ok:
                            if verify.stdout:
                                print(f"  out: {verify.stdout}", flush=True)
                            if verify.stderr:
                                print(f"  err: {verify.stderr}", flush=True)
                else:
                    for result in results[: args.top]:
                        row = {
                            "model_path": model_path,
                            "seq_len": seq_len,
                            "pattern_len": pattern_len,
                            "seed": seed,
                            "layer": result["layer"],
                            "head": result["head"],
                            "target": result.get("target", ""),
                            "negative": result.get("negative", ""),
                            "score_mode": args.score_mode,
                            "score": result.get("score", ""),
                            "prev_mean": result.get("prev_mean", ""),
                            "prev_median": result.get("prev_median", ""),
                            "prev_top1_frac": result.get("prev_top1_frac", ""),
                            "copy_mean": result.get("copy_mean", ""),
                            "copy_weighted_mean": result.get("copy_weighted_mean", ""),
                            "approx_logit_lb": result.get("logit_lb", ""),
                            "approx_eps": result.get("eps", ""),
                            "approx_margin": result.get("margin", ""),
                            "approx_min_prev": result.get("min_prev", ""),
                            "approx_value_range": result.get("value_range", ""),
                            "active": result.get("active", ""),
                            "period": period if period is not None else "",
                            "verify_status": "",
                            "verify_logit_lb": "",
                        }
                        write_csv_row(args.output, row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
