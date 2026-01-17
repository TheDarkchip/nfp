#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build an induction-head certificate from an NFP_BINARY_V1 model.

This is untrusted and uses floating-point arithmetic to produce a rational
induction-head certificate compatible with `nfp induction certify`.
"""

from __future__ import annotations

import argparse
import math
import struct
from fractions import Fraction
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def rat_from_float(x: float, decimals: int) -> Fraction:
    scale = 10 ** decimals
    return Fraction(int(round(x * scale)), scale)


def rat_to_str(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def parse_header(f) -> Dict[str, str]:
    header: Dict[str, str] = {}
    magic = f.readline().decode("ascii").strip()
    if magic != "NFP_BINARY_V1":
        raise SystemExit(f"Unsupported magic header: {magic}")
    while True:
        line = f.readline()
        if line == b"":
            raise SystemExit("Unexpected EOF while reading header.")
        text = line.decode("ascii").strip()
        if text == "BINARY_START":
            break
        if "=" in text:
            key, value = text.split("=", 1)
            header[key.strip()] = value.strip()
    return header


def read_i32(f, count: int) -> np.ndarray:
    raw = f.read(count * 4)
    if len(raw) != count * 4:
        raise SystemExit("Unexpected EOF while reading int32 payload.")
    return np.frombuffer(raw, dtype="<i4").astype(np.int64, copy=False)


def read_f64(f, count: int) -> np.ndarray:
    raw = f.read(count * 8)
    if len(raw) != count * 8:
        raise SystemExit("Unexpected EOF while reading float64 payload.")
    return np.frombuffer(raw, dtype="<f8").astype(np.float64, copy=False)


def skip_f64(f, count: int) -> None:
    offset = count * 8
    f.seek(offset, 1)


def build_prev(tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    prev = np.zeros_like(tokens)
    active = np.zeros_like(tokens, dtype=bool)
    last_seen: Dict[int, int] = {}
    for idx, tok in enumerate(tokens.tolist()):
        if idx == 0:
            prev[idx] = 0
            active[idx] = False
        else:
            if tok in last_seen:
                prev[idx] = last_seen[tok]
                active[idx] = True
            else:
                prev[idx] = 0
                active[idx] = False
        last_seen[tok] = idx
    return prev, active


def read_head_weights(
    f,
    num_layers: int,
    num_heads: int,
    model_dim: int,
    head_dim: int,
    hidden_dim: int,
    layer: int,
    head: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    target = (layer, head)
    wq = wk = wv = wo = None
    bq = bk = bv = None
    attn_bias = ln1_gamma = ln1_beta = None
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            wq_block = read_f64(f, model_dim * head_dim).reshape(model_dim, head_dim)
            bq_block = read_f64(f, head_dim)
            wk_block = read_f64(f, model_dim * head_dim).reshape(model_dim, head_dim)
            bk_block = read_f64(f, head_dim)
            wv_block = read_f64(f, model_dim * head_dim).reshape(model_dim, head_dim)
            bv_block = read_f64(f, head_dim)
            wo_block = read_f64(f, head_dim * model_dim).reshape(head_dim, model_dim)
            if (layer_idx, head_idx) == target:
                wq = wq_block
                wk = wk_block
                wv = wv_block
                wo = wo_block
                bq = bq_block
                bk = bk_block
                bv = bv_block
        attn_bias_block = read_f64(f, model_dim)
        skip_f64(f, model_dim * hidden_dim)
        skip_f64(f, hidden_dim)
        skip_f64(f, hidden_dim * model_dim)
        skip_f64(f, model_dim)
        ln1_gamma_block = read_f64(f, model_dim)
        ln1_beta_block = read_f64(f, model_dim)
        skip_f64(f, model_dim)
        skip_f64(f, model_dim)
        if layer_idx == layer:
            attn_bias = attn_bias_block
            ln1_gamma = ln1_gamma_block
            ln1_beta = ln1_beta_block
    if (
        wq is None
        or wk is None
        or wv is None
        or wo is None
        or bq is None
        or bk is None
        or bv is None
        or attn_bias is None
        or ln1_gamma is None
        or ln1_beta is None
    ):
        raise SystemExit("Failed to locate head weights.")
    return wq, bq, wk, bk, wv, bv, wo, attn_bias, ln1_gamma, ln1_beta


def read_unembed_columns(
    f,
    start: int,
    model_dim: int,
    vocab_size: int,
    target: int,
    negative: int,
) -> Tuple[np.ndarray, np.ndarray]:
    row_bytes = vocab_size * 8
    col_t = np.zeros(model_dim, dtype=np.float64)
    col_n = np.zeros(model_dim, dtype=np.float64)
    for row in range(model_dim):
        base = start + row * row_bytes
        f.seek(base + target * 8)
        col_t[row] = struct.unpack("<d", f.read(8))[0]
        f.seek(base + negative * 8)
        col_n[row] = struct.unpack("<d", f.read(8))[0]
    return col_t, col_n


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat * gamma + beta


def softmax(scores: np.ndarray) -> np.ndarray:
    shift = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shift)
    return exp / exp.sum(axis=1, keepdims=True)


def write_induction_cert(path: Path, seq: int, prev: np.ndarray,
                        scores_rat, weights_rat, eps: Fraction,
                        margin: Fraction, active_positions,
                        eps_at, weight_bound_at, vals_rat,
                        direction_target: int | None,
                        direction_negative: int | None) -> None:
    lo = min(vals_rat)
    hi = max(vals_rat)
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {seq}\n")
        if direction_target is not None and direction_negative is not None:
            f.write(f"direction-target {direction_target}\n")
            f.write(f"direction-negative {direction_negative}\n")
        f.write(f"eps {rat_to_str(eps)}\n")
        f.write(f"margin {rat_to_str(margin)}\n")
        for q in active_positions:
            f.write(f"active {q}\n")
        for q, k in enumerate(prev.tolist()):
            f.write(f"prev {q} {k}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"score {q} {k} {rat_to_str(scores_rat[q][k])}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"weight {q} {k} {rat_to_str(weights_rat[q][k])}\n")
        for q in range(seq):
            f.write(f"eps-at {q} {rat_to_str(eps_at[q])}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"weight-bound {q} {k} {rat_to_str(weight_bound_at[q][k])}\n")
        f.write(f"lo {rat_to_str(lo)}\n")
        f.write(f"hi {rat_to_str(hi)}\n")
        for k, val in enumerate(vals_rat):
            val_str = rat_to_str(val)
            f.write(f"val {k} {val_str}\n")
            f.write(f"val-lo {k} {val_str}\n")
            f.write(f"val-hi {k} {val_str}\n")


def write_value_range(path: Path, seq: int, values, decimals: int,
                      direction_target=None, direction_negative=None) -> None:
    vals_rat = [rat_from_float(float(values[k]), decimals) for k in range(seq)]
    lo = min(vals_rat)
    hi = max(vals_rat)
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {seq}\n")
        if direction_target is not None and direction_negative is not None:
            f.write(f"direction-target {direction_target}\n")
            f.write(f"direction-negative {direction_negative}\n")
        f.write(f"lo {rat_to_str(lo)}\n")
        f.write(f"hi {rat_to_str(hi)}\n")
        for k, val in enumerate(vals_rat):
            f.write(f"val {k} {rat_to_str(val)}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="Path to NFP_BINARY_V1 model")
    ap.add_argument("--layer", type=int, required=True, help="Layer index")
    ap.add_argument("--head", type=int, required=True, help="Head index")
    ap.add_argument("--output", type=Path, required=True,
                    help="Path for induction-head certificate")
    ap.add_argument("--values-out", type=Path,
                    help="Optional path for a value-range certificate")
    ap.add_argument("--direction-target", type=int, required=True,
                    help="Target token id for logit-diff direction")
    ap.add_argument("--direction-negative", type=int, required=True,
                    help="Negative token id for logit-diff direction")
    ap.add_argument("--decimals", type=int, default=6,
                    help="Decimal rounding for rationals")
    ap.add_argument("--active-eps-max", default="1/2",
                    help="Maximum eps to include an active position")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Missing model file: {args.model}")

    with args.model.open("rb") as f:
        header = parse_header(f)
        num_layers = int(header["num_layers"])
        num_heads = int(header["num_heads"])
        model_dim = int(header["model_dim"])
        head_dim = int(header["head_dim"])
        vocab_size = int(header["vocab_size"])
        seq_len = int(header["seq_len"])
        hidden_dim = int(header["hidden_dim"])
        ln_eps = float(header.get("layer_norm_eps", header.get("eps", "0")))

        if args.layer < 0 or args.layer >= num_layers:
            raise SystemExit("layer index out of range")
        if args.head < 0 or args.head >= num_heads:
            raise SystemExit("head index out of range")
        if not (0 <= args.direction_target < vocab_size):
            raise SystemExit("direction-target out of vocab range")
        if not (0 <= args.direction_negative < vocab_size):
            raise SystemExit("direction-negative out of vocab range")

        tokens = read_i32(f, seq_len)
        embeddings = read_f64(f, seq_len * model_dim).reshape(seq_len, model_dim)

        wq, bq, wk, bk, wv, bv, wo_raw, _attn_bias, ln1_gamma, ln1_beta = read_head_weights(
            f,
            num_layers,
            num_heads,
            model_dim,
            head_dim,
            hidden_dim,
            args.layer,
            args.head,
        )

        skip_f64(f, model_dim)
        skip_f64(f, model_dim)

        unembed_start = f.tell()
        col_target, col_negative = read_unembed_columns(
            f,
            unembed_start,
            model_dim,
            vocab_size,
            args.direction_target,
            args.direction_negative,
        )

    prev, active_mask = build_prev(tokens)
    candidate_positions = [int(i) for i, flag in enumerate(active_mask) if flag]
    active_eps_max = Fraction(args.active_eps_max)

    scale_denom = int(math.isqrt(head_dim))
    if scale_denom * scale_denom != head_dim:
        scale = 1.0 / math.sqrt(head_dim)
    else:
        scale = 1.0 / scale_denom

    ln = layer_norm(embeddings, ln1_gamma, ln1_beta, ln_eps)
    q = ln @ wq + bq
    k = ln @ wk + bk
    v = ln @ wv + bv

    scores = scale * (q @ k.T)
    mask_value = -10000.0
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores = scores.copy()
    scores[mask] = mask_value
    weights = softmax(scores)

    scores_rat = [[rat_from_float(float(scores[q, k]), args.decimals)
                   for k in range(seq_len)] for q in range(seq_len)]
    weights_rat = [[rat_from_float(float(weights[q, k]), args.decimals)
                    for k in range(seq_len)] for q in range(seq_len)]

    for q in range(seq_len):
        total = sum(weights_rat[q], Fraction(0))
        if total == 0:
            raise SystemExit(f"zero weight sum at q={q}")
        weights_rat[q] = [w / total for w in weights_rat[q]]

    eps_by_q: dict[int, Fraction] = {}
    margin_by_q: dict[int, Fraction] = {}
    for q in range(seq_len):
        prev_q = prev[q]
        prev_w = weights_rat[q][prev_q]
        if seq_len == 1:
            max_other = Fraction(0)
        else:
            max_other = max(weights_rat[q][k] for k in range(seq_len) if k != prev_q)
        deficit = Fraction(1) - prev_w
        eps_by_q[q] = max(max_other, deficit)

        diffs = [scores_rat[q][prev_q] - scores_rat[q][k]
                 for k in range(seq_len) if k != prev_q]
        margin_by_q[q] = min(diffs) if diffs else Fraction(0)

    active_positions = [q for q in candidate_positions if eps_by_q[q] <= active_eps_max]
    if not active_positions and candidate_positions:
        print("Warning: no active positions satisfy active-eps-max; certificate may be vacuous.")

    if not active_positions and seq_len > 1:
        if candidate_positions:
            print("Warning: no active positions satisfy active-eps-max; using all nonzero queries.")
        active_positions = list(range(1, seq_len))
    if active_positions:
        eps = max(eps_by_q[q] for q in active_positions)
        margin = min((margin_by_q[q] for q in active_positions), default=Fraction(0))
    else:
        eps = Fraction(0)
        margin = Fraction(0)

    eps_at = []
    for q in range(seq_len):
        prev_q = prev[q]
        if seq_len == 1:
            max_other = Fraction(0)
        else:
            max_other = max(weights_rat[q][k] for k in range(seq_len) if k != prev_q)
        deficit = Fraction(1) - weights_rat[q][prev_q]
        eps_at.append(max(max_other, deficit))
    weight_bound_at = weights_rat

    wo = wo_raw.T
    direction = col_target - col_negative
    dir_head = wo.T @ direction
    dir_vals = v @ dir_head
    vals_rat = [rat_from_float(float(dir_vals[k]), args.decimals) for k in range(seq_len)]

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_induction_cert(
        output_path,
        seq_len,
        prev,
        scores_rat,
        weights_rat,
        eps,
        margin,
        active_positions,
        eps_at,
        weight_bound_at,
        vals_rat,
        args.direction_target,
        args.direction_negative,
    )

    if args.values_out:
        values_path = args.values_out
        values_path.parent.mkdir(parents=True, exist_ok=True)
        write_value_range(values_path, seq_len, dir_vals, args.decimals,
                          direction_target=args.direction_target,
                          direction_negative=args.direction_negative)
        print(f"Wrote value-range certificate to {values_path}")

    print(f"Wrote induction-head certificate to {output_path}")
    if candidate_positions:
        print(f"Active positions: {len(active_positions)}/{len(candidate_positions)}")


if __name__ == "__main__":
    main()
