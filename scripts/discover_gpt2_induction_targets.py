#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Discover promising GPT-2 induction heads and logit-diff directions from an NFP binary.

This script is untrusted: it uses floating-point arithmetic to score candidates
and optionally invokes the Lean verifier (`nfp induction certify_head_model_nonvacuous`)
to confirm nonvacuous bounds when scoring by logit-diff.

Layer/head indices are one-based to align with the mechanistic interpretability
literature.

By default, `prev`/active are built from bigram prefix matches (the token at
q-1 maps to the *following* token after its previous occurrence), and heads are
ranked by attention to `prev`. Use `--score-mode=copy` or `--score-mode=attn_copy`
to include OV/copying alignment in the ranking. Use `--use-activations` to
score heads using real layer activations from a HuggingFace GPT-2 model rather
than the embedding-only approximation stored in the NFP file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class HeadResult:
    layer: int
    head: int
    target: int
    negative: int
    logit_lb: float
    eps: float
    margin: float
    min_prev: float
    value_range: float
    active: int
    prev_mean: float
    prev_median: float
    prev_top1_frac: float


@dataclass(frozen=True)
class AttnResult:
    layer: int
    head: int
    score: float
    prev_mean: float
    prev_median: float
    prev_top1_frac: float
    copy_mean: float
    copy_weighted_mean: float
    eps: float
    margin: float
    active: int


@dataclass(frozen=True)
class StripeResult:
    layer: int
    head: int
    score: float
    stripe_mean: float
    stripe_median: float
    stripe_top1_frac: float
    eps: float
    margin: float
    active: int


@dataclass(frozen=True)
class CircuitResult:
    prev_layer: int
    prev_head: int
    induction_layer: int
    induction_head: int
    score: float
    prev_mean: float
    prev_median: float
    prev_top1_frac: float
    stripe_mean: float
    stripe_median: float
    stripe_top1_frac: float


@dataclass(frozen=True)
class CircuitCopyResult:
    prev_layer: int
    prev_head: int
    induction_layer: int
    induction_head: int
    score: float
    prev_mean: float
    prev_median: float
    prev_top1_frac: float
    stripe_mean: float
    stripe_median: float
    stripe_top1_frac: float
    copy_mean: float
    copy_weighted_mean: float


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


def build_prev_bigram(tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bigram-prefix induction prev: for each position q>=1, look at token at q-1,
    find its previous occurrence index j, and set prev[q] = j + 1.

    Example (tokens = [1,2,1,3,2,1]):
      prev = [0,0,0,1,0,2], active = [F,F,F,T,F,T]
    """
    prev_token, active_token = build_prev(tokens)
    prev = np.zeros_like(tokens)
    active = np.zeros_like(tokens, dtype=bool)
    if tokens.size <= 1:
        return prev, active
    prev_shift = prev_token[:-1] + 1
    active_shift = active_token[:-1]
    prev[1:] = np.where(active_shift, prev_shift, 0)
    active[1:] = active_shift
    return prev, active


def build_prev_period(seq_len: int, period: int) -> Tuple[np.ndarray, np.ndarray]:
    prev = np.zeros(seq_len, dtype=np.int64)
    active = np.zeros(seq_len, dtype=bool)
    idx = np.arange(seq_len)
    mask = idx >= period
    prev[mask] = idx[mask] - period
    active[mask] = True
    return prev, active


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat * gamma + beta


def skip_head_weights(f, model_dim: int, head_dim: int) -> None:
    skip_f64(f, model_dim * head_dim)  # wq
    skip_f64(f, head_dim)  # bq
    skip_f64(f, model_dim * head_dim)  # wk
    skip_f64(f, head_dim)  # bk
    skip_f64(f, model_dim * head_dim)  # wv
    skip_f64(f, head_dim)  # bv
    skip_f64(f, head_dim * model_dim)  # wo


def skip_layer_weights(
    f,
    model_dim: int,
    head_dim: int,
    num_heads: int,
    hidden_dim: int,
) -> None:
    for _ in range(num_heads):
        skip_head_weights(f, model_dim, head_dim)
    skip_f64(f, model_dim)  # attn bias
    skip_f64(f, model_dim * hidden_dim)
    skip_f64(f, hidden_dim)
    skip_f64(f, hidden_dim * model_dim)
    skip_f64(f, model_dim)
    skip_f64(f, model_dim)  # ln1 gamma
    skip_f64(f, model_dim)  # ln1 beta
    skip_f64(f, model_dim)  # ln2 gamma
    skip_f64(f, model_dim)  # ln2 beta


def load_hf_model_and_states(tokens: np.ndarray, model_name: str, device: str):
    try:
        import torch
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Activation mode requires torch + transformers. "
            "Install them (e.g., `uv run --with torch --with transformers ...`)."
        ) from exc

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise SystemExit("HuggingFace model did not return hidden states.")
    return model, hidden_states


def get_transformer_blocks(model):
    if hasattr(model, "transformer"):
        return model.transformer.h
    if hasattr(model, "h"):
        return model.h
    raise SystemExit("Unsupported HuggingFace model structure (missing transformer blocks).")


def compute_head_data_from_activations(
    model,
    hidden_states,
    layers: List[int],
    heads: List[int],
    prev: np.ndarray,
    active_positions: List[int],
    prev_indices: np.ndarray,
    head_dim: int,
    seq_len: int,
    stripe_prev: np.ndarray | None = None,
    stripe_positions: List[int] | None = None,
    prevtok_prev: np.ndarray | None = None,
    prevtok_positions: List[int] | None = None,
) -> Tuple[
    Dict[
        Tuple[int, int],
        Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float],
    ],
    Dict[Tuple[int, int], Tuple[float, float, float, float, float]],
    Dict[Tuple[int, int], Tuple[float, float, float, float, float]],
]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Activation mode requires torch.") from exc

    def split_heads(x: "torch.Tensor", num_heads: int, head_dim_local: int) -> "torch.Tensor":
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, num_heads, head_dim_local)
        return x.permute(0, 2, 1, 3)

    blocks = get_transformer_blocks(model)
    head_data: Dict[
        Tuple[int, int],
        Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float],
    ] = {}
    stripe_data: Dict[Tuple[int, int], Tuple[float, float, float, float, float]] = {}
    prevtok_data: Dict[Tuple[int, int], Tuple[float, float, float, float, float]] = {}
    device = hidden_states[0].device
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
        diagonal=1,
    )
    for layer_idx in layers:
        block = blocks[layer_idx]
        hidden = hidden_states[layer_idx]
        ln = block.ln_1(hidden)
        qkv = block.attn.c_attn(ln)
        split_size = getattr(block.attn, "split_size", qkv.shape[-1] // 3)
        q, k, v = qkv.split(split_size, dim=2)
        num_heads = getattr(block.attn, "num_heads", q.shape[-1] // head_dim)
        head_dim_local = getattr(block.attn, "head_dim", head_dim)
        if head_dim_local != head_dim:
            raise SystemExit("HuggingFace head_dim does not match NFP header.")
        scale = 1.0 / math.sqrt(head_dim_local)
        q = split_heads(q, num_heads, head_dim_local)
        k = split_heads(k, num_heads, head_dim_local)
        v = split_heads(v, num_heads, head_dim_local)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(causal_mask, -10000.0)
        weights = torch.softmax(scores, dim=-1)
        wo_full = block.attn.c_proj.weight

        for head_idx in heads:
            weights_head = weights[0, head_idx]
            scores_head = scores[0, head_idx]
            weights_np = weights_head.detach().cpu().numpy()
            scores_np = scores_head.detach().cpu().numpy()
            eps, margin, prev_mean, prev_median, prev_top1 = compute_eps_margin(
                weights_np, scores_np, prev, active_positions
            )
            if stripe_prev is not None and stripe_positions is not None:
                eps_s, margin_s, stripe_mean, stripe_median, stripe_top1 = compute_eps_margin(
                    weights_np, scores_np, stripe_prev, stripe_positions
                )
                stripe_data[(layer_idx, head_idx)] = (
                    stripe_mean,
                    stripe_median,
                    stripe_top1,
                    eps_s,
                    margin_s,
                )
            if prevtok_prev is not None and prevtok_positions is not None:
                eps_p, margin_p, prevtok_mean, prevtok_median, prevtok_top1 = compute_eps_margin(
                    weights_np, scores_np, prevtok_prev, prevtok_positions
                )
                prevtok_data[(layer_idx, head_idx)] = (
                    prevtok_mean,
                    prevtok_median,
                    prevtok_top1,
                    eps_p,
                    margin_p,
                )
            prev_weights = weights_np[np.array(active_positions), prev_indices]

            v_head = v[0, head_idx]
            v_np = v_head.detach().cpu().numpy()
            start = head_idx * head_dim_local
            end = start + head_dim_local
            wo_np = wo_full[start:end, :].detach().cpu().numpy()

            head_data[(layer_idx, head_idx)] = (
                v_np,
                wo_np,
                prev_weights,
                eps,
                margin,
                prev_mean,
                prev_median,
                prev_top1,
            )
    return head_data, stripe_data, prevtok_data


def softmax(scores: np.ndarray) -> np.ndarray:
    shift = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shift)
    return exp / exp.sum(axis=1, keepdims=True)


def parse_index_list(raw: str | None, max_value: int) -> List[int] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw.lower() == "all":
        return list(range(max_value))
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx <= 0 or idx > max_value:
            raise ValueError(f"index {idx} out of range [1,{max_value}]")
        out.append(idx - 1)
    return out


def resolve_nfp_cmd(nfp_bin: str | None) -> List[str]:
    if nfp_bin:
        return [nfp_bin]
    env_bin = os.environ.get("NFP_BIN")
    if env_bin:
        return [env_bin]
    local_bin = Path(".lake/build/bin/nfp")
    if local_bin.exists():
        return [str(local_bin)]
    return ["lake", "exe", "nfp"]


def read_unembed_column(
    f,
    start: int,
    model_dim: int,
    vocab_size: int,
    col: int,
) -> np.ndarray:
    if col < 0 or col >= vocab_size:
        raise ValueError(f"column {col} out of range")
    row_bytes = vocab_size * 8
    data = np.zeros(model_dim, dtype=np.float64)
    for row in range(model_dim):
        base = start + row * row_bytes
        f.seek(base + col * 8)
        data[row] = np.frombuffer(f.read(8), dtype="<f8")[0]
    return data


def compute_eps_margin(
    weights: np.ndarray,
    scores: np.ndarray,
    prev: np.ndarray,
    active_positions: Iterable[int],
) -> Tuple[float, float, float, float, float]:
    eps_vals: List[float] = []
    margin_vals: List[float] = []
    prev_vals: List[float] = []
    max_other_vals: List[float] = []
    for q in active_positions:
        prev_q = int(prev[q])
        prev_w = weights[q, prev_q]
        max_other = np.max(np.delete(weights[q], prev_q))
        eps_vals.append(max(max_other, 1.0 - prev_w))
        diffs = scores[q, prev_q] - np.delete(scores[q], prev_q)
        margin_vals.append(float(np.min(diffs)) if diffs.size > 0 else 0.0)
        prev_vals.append(float(prev_w))
        max_other_vals.append(float(max_other))
    if not eps_vals:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    prev_arr = np.asarray(prev_vals, dtype=np.float64)
    max_other_arr = np.asarray(max_other_vals, dtype=np.float64)
    prev_mean = float(prev_arr.mean())
    prev_median = float(np.median(prev_arr))
    prev_top1 = float(np.mean(prev_arr >= max_other_arr))
    return max(eps_vals), min(margin_vals), prev_mean, prev_median, prev_top1


def compute_copy_scores(
    ov: np.ndarray,
    weights_prev: np.ndarray,
    columns: Dict[int, np.ndarray],
    tokens: np.ndarray,
    prev: np.ndarray,
    active_positions: Iterable[int],
) -> Tuple[float, float]:
    copy_vals: List[float] = []
    copy_weighted_vals: List[float] = []
    for idx, q in enumerate(active_positions):
        tok = int(tokens[q])
        col = columns.get(tok)
        if col is None:
            continue
        prev_q = int(prev[q])
        val = float(ov[prev_q] @ col)
        copy_vals.append(val)
        copy_weighted_vals.append(float(weights_prev[idx]) * val)
    if not copy_vals:
        return 0.0, 0.0
    return float(np.mean(copy_vals)), float(np.mean(copy_weighted_vals))


def format_result(result: HeadResult) -> str:
    layer = result.layer + 1
    head = result.head + 1
    return (
        f"L{layer}H{head} target={result.target} "
        f"negative={result.negative} logitLB={result.logit_lb:.6f} "
        f"eps={result.eps:.6f} margin={result.margin:.6f} "
        f"minPrev={result.min_prev:.6f} range={result.value_range:.6f} "
        f"prevMean={result.prev_mean:.6f} prevMedian={result.prev_median:.6f} "
        f"prevTop1={result.prev_top1_frac:.3f} active={result.active}"
    )


def format_attn_result(result: AttnResult) -> str:
    layer = result.layer + 1
    head = result.head + 1
    return (
        f"L{layer}H{head} score={result.score:.6f} "
        f"prevMean={result.prev_mean:.6f} prevMedian={result.prev_median:.6f} "
        f"prevTop1={result.prev_top1_frac:.3f} "
        f"copyMean={result.copy_mean:.6f} copyWeighted={result.copy_weighted_mean:.6f} "
        f"eps={result.eps:.6f} margin={result.margin:.6f} active={result.active}"
    )


def format_stripe_result(result: StripeResult) -> str:
    layer = result.layer + 1
    head = result.head + 1
    return (
        f"L{layer}H{head} score={result.score:.6f} "
        f"stripeMean={result.stripe_mean:.6f} stripeMedian={result.stripe_median:.6f} "
        f"stripeTop1={result.stripe_top1_frac:.3f} "
        f"eps={result.eps:.6f} margin={result.margin:.6f} active={result.active}"
    )


def format_circuit_result(result: CircuitResult) -> str:
    prev_layer = result.prev_layer + 1
    prev_head = result.prev_head + 1
    ind_layer = result.induction_layer + 1
    ind_head = result.induction_head + 1
    return (
        f"prev=L{prev_layer}H{prev_head} ind=L{ind_layer}H{ind_head} "
        f"score={result.score:.6f} prevMean={result.prev_mean:.6f} "
        f"prevMedian={result.prev_median:.6f} prevTop1={result.prev_top1_frac:.3f} "
        f"stripeMean={result.stripe_mean:.6f} stripeMedian={result.stripe_median:.6f} "
        f"stripeTop1={result.stripe_top1_frac:.3f}"
    )


def format_circuit_copy_result(result: CircuitCopyResult) -> str:
    prev_layer = result.prev_layer + 1
    prev_head = result.prev_head + 1
    ind_layer = result.induction_layer + 1
    ind_head = result.induction_head + 1
    return (
        f"prev=L{prev_layer}H{prev_head} ind=L{ind_layer}H{ind_head} "
        f"score={result.score:.6f} prevMean={result.prev_mean:.6f} "
        f"stripeMean={result.stripe_mean:.6f} copyMean={result.copy_mean:.6f} "
        f"copyWeighted={result.copy_weighted_mean:.6f} "
        f"prevMedian={result.prev_median:.6f} prevTop1={result.prev_top1_frac:.3f} "
        f"stripeMedian={result.stripe_median:.6f} stripeTop1={result.stripe_top1_frac:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Path to NFP_BINARY_V1 model")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Maximum unique tokens from the prompt to consider")
    parser.add_argument("--top", type=int, default=20, help="Number of results to report")
    parser.add_argument("--verify-top", type=int, default=0,
                        help="Run verifier on the top N candidates")
    parser.add_argument(
        "--score-mode",
        choices=["attn", "copy", "attn_copy", "stripe", "circuit", "circuit_copy", "logit"],
        default="attn",
        help=(
            "Rank heads by attention to prev (attn), OV copy score (copy), "
            "attention-weighted copy score (attn_copy), induction stripe attention "
            "(stripe), circuit pairing (circuit), circuit + copy (circuit_copy), "
            "or logit lower bound (logit)."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score threshold for the selected score mode.",
    )
    parser.add_argument(
        "--min-copy",
        type=float,
        default=None,
        help="Optional minimum OV copy score.",
    )
    parser.add_argument("--min-eps", type=float, default=0.5,
                        help="Filter candidates with eps above this value")
    parser.add_argument("--min-margin", type=float, default=0.0,
                        help="Filter candidates with margin below this value")
    parser.add_argument("--min-logit-lb", type=float, default=0.0,
                        help="Filter candidates with logit lower bound below this value")
    parser.add_argument("--layers", help="Comma-separated layer list or 'all'")
    parser.add_argument("--heads", help="Comma-separated head list or 'all'")
    parser.add_argument("--period", type=int, help="Optional prompt period override")
    parser.add_argument(
        "--prev-mode",
        choices=["bigram", "token", "period"],
        default="bigram",
        help="Choose prev/active construction (default: bigram prefix match).",
    )
    parser.add_argument(
        "--stripe-period",
        type=int,
        help="Period for induction stripe scoring (required for --score-mode=stripe).",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/gpt2_induction_discover.txt"))
    parser.add_argument("--json-out", type=Path, help="Optional JSON output path")
    parser.add_argument("--nfp-bin", help="Path to nfp binary (defaults to .lake/build/bin/nfp)")
    parser.add_argument(
        "--use-activations",
        action="store_true",
        help="Use HuggingFace GPT-2 activations for Q/K/V instead of embedding-only approximation.",
    )
    parser.add_argument(
        "--hf-model",
        default="gpt2",
        help="HuggingFace model name or path (activation mode).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for activation mode (e.g. cpu, cuda, mps).",
    )
    args = parser.parse_args()

    if args.max_tokens <= 1:
        raise SystemExit("max-tokens must be at least 2")
    if args.verify_top > 0 and args.score_mode != "logit":
        raise SystemExit("--verify-top requires --score-mode=logit")
    if args.score_mode in {"stripe", "circuit", "circuit_copy"} and args.stripe_period is None:
        raise SystemExit("--score-mode=stripe/circuit requires --stripe-period")
    if args.score_mode in {"circuit", "circuit_copy"} and args.prev_mode != "bigram":
        raise SystemExit("--score-mode=circuit requires --prev-mode=bigram")

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

        layers = parse_index_list(args.layers, num_layers) or list(range(num_layers))
        heads = parse_index_list(args.heads, num_heads) or list(range(num_heads))

        tokens = read_i32(f, seq_len)
        if args.use_activations:
            skip_f64(f, seq_len * model_dim)
            embeddings = None
        else:
            embeddings = read_f64(f, seq_len * model_dim).reshape(seq_len, model_dim)

        if args.prev_mode != "period" and args.period is not None:
            raise SystemExit("--period is incompatible with --prev-mode=token/bigram")
        if args.prev_mode == "period" and args.period is None:
            raise SystemExit("--prev-mode=period requires --period")

        if args.prev_mode == "period":
            period = int(args.period)
            prev, active_mask = build_prev_period(seq_len, period)
        elif args.prev_mode == "bigram":
            prev, active_mask = build_prev_bigram(tokens)
        else:
            prev, active_mask = build_prev(tokens)

        active_positions = [int(i) for i, flag in enumerate(active_mask) if flag]
        if not active_positions:
            raise SystemExit("No active positions found in the prompt")
        prev_indices = prev[np.array(active_positions, dtype=np.int64)]
        stripe_prev = None
        stripe_positions = None
        if args.score_mode in {"stripe", "circuit", "circuit_copy"}:
            stripe_period = int(args.stripe_period)
            stripe_prev, stripe_active = build_prev_period(seq_len, stripe_period)
            stripe_positions = [int(i) for i, flag in enumerate(stripe_active) if flag]
            if not stripe_positions:
                raise SystemExit("No stripe positions found for the requested period")
        prevtok_prev = None
        prevtok_positions = None
        if args.score_mode in {"circuit", "circuit_copy"}:
            prevtok_prev, prevtok_active = build_prev_period(seq_len, 1)
            prevtok_positions = [int(i) for i, flag in enumerate(prevtok_active) if flag]
            if not prevtok_positions:
                raise SystemExit("No previous-token positions found")

        prompt_tokens = sorted({int(tok) for tok in tokens.tolist()})
        unique_tokens = []
        seen = set()
        for tok in tokens.tolist():
            if tok not in seen:
                seen.add(tok)
                unique_tokens.append(int(tok))
            if len(unique_tokens) >= args.max_tokens:
                break
        if len(unique_tokens) < 2:
            raise SystemExit("Need at least two unique tokens to form directions")

        head_data: Dict[
            Tuple[int, int],
            Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float],
        ] = {}
        stripe_data: Dict[Tuple[int, int], Tuple[float, float, float, float, float]] = {}
        prevtok_data: Dict[Tuple[int, int], Tuple[float, float, float, float, float]] = {}

        hf_model = None
        hf_states = None
        if args.use_activations:
            hf_model, hf_states = load_hf_model_and_states(tokens, args.hf_model, args.device)
            config = getattr(hf_model, "config", None)
            if config is not None:
                if getattr(config, "n_layer", num_layers) != num_layers:
                    raise SystemExit("HuggingFace model layer count does not match NFP header.")
                if getattr(config, "n_head", num_heads) != num_heads:
                    raise SystemExit("HuggingFace model head count does not match NFP header.")
                if getattr(config, "n_embd", model_dim) != model_dim:
                    raise SystemExit("HuggingFace model dimension does not match NFP header.")
                if getattr(config, "vocab_size", vocab_size) != vocab_size:
                    raise SystemExit("HuggingFace vocab size does not match NFP header.")
                if getattr(config, "n_positions", seq_len) < seq_len:
                    raise SystemExit("Prompt length exceeds HuggingFace model context.")
            if len(hf_states) < num_layers + 1:
                raise SystemExit("Hidden state count is smaller than expected.")
            if hf_states[0].shape[1] != seq_len:
                raise SystemExit("Hidden state sequence length does not match NFP header.")
        for layer_idx in range(num_layers):
            if args.use_activations:
                skip_layer_weights(f, model_dim, head_dim, num_heads, hidden_dim)
                continue
            head_weights = []
            for _ in range(num_heads):
                wq = read_f64(f, model_dim * head_dim).reshape(model_dim, head_dim)
                bq = read_f64(f, head_dim)
                wk = read_f64(f, model_dim * head_dim).reshape(model_dim, head_dim)
                bk = read_f64(f, head_dim)
                wv = read_f64(f, model_dim * head_dim).reshape(model_dim, head_dim)
                bv = read_f64(f, head_dim)
                wo = read_f64(f, head_dim * model_dim).reshape(head_dim, model_dim)
                head_weights.append((wq, bq, wk, bk, wv, bv, wo))
            _attn_bias = read_f64(f, model_dim)
            skip_f64(f, model_dim * hidden_dim)
            skip_f64(f, hidden_dim)
            skip_f64(f, hidden_dim * model_dim)
            skip_f64(f, model_dim)
            ln1_gamma = read_f64(f, model_dim)
            ln1_beta = read_f64(f, model_dim)
            skip_f64(f, model_dim)
            skip_f64(f, model_dim)

            if layer_idx not in layers:
                continue

            ln = layer_norm(embeddings, ln1_gamma, ln1_beta, ln_eps)
            scale = 1.0 / np.sqrt(head_dim)
            for head_idx in heads:
                wq, bq, wk, bk, wv, bv, wo = head_weights[head_idx]
                q = ln @ wq + bq
                k = ln @ wk + bk
                v = ln @ wv + bv

                scores = scale * (q @ k.T)
                mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
                scores = scores.copy()
                scores[mask] = -10000.0
                weights = softmax(scores)

                eps, margin, prev_mean, prev_median, prev_top1 = compute_eps_margin(
                    weights, scores, prev, active_positions
                )
                if stripe_prev is not None and stripe_positions is not None:
                    eps_s, margin_s, stripe_mean, stripe_median, stripe_top1 = compute_eps_margin(
                        weights, scores, stripe_prev, stripe_positions
                    )
                    stripe_data[(layer_idx, head_idx)] = (
                        stripe_mean,
                        stripe_median,
                        stripe_top1,
                        eps_s,
                        margin_s,
                    )
                if prevtok_prev is not None and prevtok_positions is not None:
                    eps_p, margin_p, prevtok_mean, prevtok_median, prevtok_top1 = compute_eps_margin(
                        weights, scores, prevtok_prev, prevtok_positions
                    )
                    prevtok_data[(layer_idx, head_idx)] = (
                        prevtok_mean,
                        prevtok_median,
                        prevtok_top1,
                        eps_p,
                        margin_p,
                    )
                prev_weights = weights[np.array(active_positions), prev_indices]
                head_data[(layer_idx, head_idx)] = (
                    v,
                    wo,
                    prev_weights,
                    eps,
                    margin,
                    prev_mean,
                    prev_median,
                    prev_top1,
                )

        ln_f_gamma = read_f64(f, model_dim)
        _ln_f_beta = read_f64(f, model_dim)
        _ = ln_f_gamma
        unembed_start = f.tell()

        columns: Dict[int, np.ndarray] = {}
        for tok in prompt_tokens:
            columns[tok] = read_unembed_column(
                f,
                unembed_start,
                model_dim,
                vocab_size,
                tok,
            )

        if args.use_activations:
            head_data, stripe_data, prevtok_data = compute_head_data_from_activations(
                hf_model,
                hf_states,
                layers,
                heads,
                prev,
                active_positions,
                prev_indices,
                head_dim,
                seq_len,
                stripe_prev=stripe_prev,
                stripe_positions=stripe_positions,
                prevtok_prev=prevtok_prev,
                prevtok_positions=prevtok_positions,
            )

    results: List[HeadResult] = []
    attn_results: List[AttnResult] = []
    stripe_results: List[StripeResult] = []
    circuit_results: List[CircuitResult] = []
    circuit_copy_results: List[CircuitCopyResult] = []
    for (layer_idx, head_idx), (
        v,
        wo,
        prev_weights,
        eps,
        margin,
        prev_mean,
        prev_median,
        prev_top1,
    ) in head_data.items():
        if args.score_mode == "logit":
            if eps > args.min_eps or margin < args.min_margin:
                continue
        if args.score_mode == "stripe":
            stripe = stripe_data.get((layer_idx, head_idx))
            if stripe is None:
                continue
            stripe_mean, stripe_median, stripe_top1, eps_s, margin_s = stripe
            score = stripe_mean
            if score < args.min_score:
                continue
            stripe_results.append(
                StripeResult(
                    layer=layer_idx,
                    head=head_idx,
                    score=score,
                    stripe_mean=stripe_mean,
                    stripe_median=stripe_median,
                    stripe_top1_frac=stripe_top1,
                    eps=eps_s,
                    margin=margin_s,
                    active=len(stripe_positions) if stripe_positions is not None else 0,
                )
            )
            continue
        if args.score_mode == "circuit":
            stripe = stripe_data.get((layer_idx, head_idx))
            if stripe is None:
                continue
            stripe_mean, stripe_median, stripe_top1, _eps_s, _margin_s = stripe
            best_prev: CircuitResult | None = None
            for (prev_layer, prev_head), prev_stats in prevtok_data.items():
                if prev_layer >= layer_idx:
                    continue
                prev_mean, prev_median, prev_top1, _eps_p, _margin_p = prev_stats
                score = prev_mean * stripe_mean
                if score < args.min_score:
                    continue
                candidate = CircuitResult(
                    prev_layer=prev_layer,
                    prev_head=prev_head,
                    induction_layer=layer_idx,
                    induction_head=head_idx,
                    score=score,
                    prev_mean=prev_mean,
                    prev_median=prev_median,
                    prev_top1_frac=prev_top1,
                    stripe_mean=stripe_mean,
                    stripe_median=stripe_median,
                    stripe_top1_frac=stripe_top1,
                )
                if best_prev is None or candidate.score > best_prev.score:
                    best_prev = candidate
            if best_prev is not None:
                circuit_results.append(best_prev)
            continue
        if args.score_mode == "circuit_copy":
            stripe = stripe_data.get((layer_idx, head_idx))
            if stripe is None:
                continue
            stripe_mean, stripe_median, stripe_top1, _eps_s, _margin_s = stripe
            ov = v @ wo
            copy_mean, copy_weighted_mean = compute_copy_scores(
                ov,
                prev_weights,
                columns,
                tokens,
                prev,
                active_positions,
            )
            if args.min_copy is not None and copy_mean < args.min_copy:
                continue
            copy_score = max(copy_mean, 0.0)
            best_prev: CircuitCopyResult | None = None
            for (prev_layer, prev_head), prev_stats in prevtok_data.items():
                if prev_layer >= layer_idx:
                    continue
                prev_mean, prev_median, prev_top1, _eps_p, _margin_p = prev_stats
                score = prev_mean * stripe_mean * copy_score
                if score < args.min_score:
                    continue
                candidate = CircuitCopyResult(
                    prev_layer=prev_layer,
                    prev_head=prev_head,
                    induction_layer=layer_idx,
                    induction_head=head_idx,
                    score=score,
                    prev_mean=prev_mean,
                    prev_median=prev_median,
                    prev_top1_frac=prev_top1,
                    stripe_mean=stripe_mean,
                    stripe_median=stripe_median,
                    stripe_top1_frac=stripe_top1,
                    copy_mean=copy_mean,
                    copy_weighted_mean=copy_weighted_mean,
                )
                if best_prev is None or candidate.score > best_prev.score:
                    best_prev = candidate
            if best_prev is not None:
                circuit_copy_results.append(best_prev)
            continue
        if args.score_mode != "logit":
            ov = v @ wo
            copy_mean, copy_weighted_mean = compute_copy_scores(
                ov,
                prev_weights,
                columns,
                tokens,
                prev,
                active_positions,
            )
            if args.min_copy is not None and copy_mean < args.min_copy:
                continue
            if args.score_mode == "attn":
                score = prev_mean
            elif args.score_mode == "copy":
                score = copy_mean
            else:
                score = copy_weighted_mean
            if score < args.min_score:
                continue
            attn_results.append(
                AttnResult(
                    layer=layer_idx,
                    head=head_idx,
                    score=score,
                    prev_mean=prev_mean,
                    prev_median=prev_median,
                    prev_top1_frac=prev_top1,
                    copy_mean=copy_mean,
                    copy_weighted_mean=copy_weighted_mean,
                    eps=eps,
                    margin=margin,
                    active=len(active_positions),
                )
            )
            continue
        proj: Dict[int, np.ndarray] = {}
        for tok in unique_tokens:
            dir_head = wo @ columns[tok]
            proj[tok] = v @ dir_head
        best: HeadResult | None = None
        for target in unique_tokens:
            vals_target = proj[target]
            for negative in unique_tokens:
                if target == negative:
                    continue
                vals = vals_target - proj[negative]
                vals_prev = vals[prev_indices]
                min_prev = float(vals_prev.min()) if vals_prev.size else 0.0
                value_range = float(vals.max() - vals.min())
                logit_lb = min_prev - eps * value_range
                if logit_lb < args.min_logit_lb:
                    continue
                candidate = HeadResult(
                    layer=layer_idx,
                    head=head_idx,
                    target=target,
                    negative=negative,
                    logit_lb=logit_lb,
                    eps=eps,
                    margin=margin,
                    min_prev=min_prev,
                    value_range=value_range,
                    active=len(active_positions),
                    prev_mean=prev_mean,
                    prev_median=prev_median,
                    prev_top1_frac=prev_top1,
                )
                if best is None or candidate.logit_lb > best.logit_lb:
                    best = candidate
        if best is not None:
            results.append(best)

    if args.score_mode == "circuit_copy":
        circuit_copy_results.sort(key=lambda r: r.score, reverse=True)
    elif args.score_mode == "circuit":
        circuit_results.sort(key=lambda r: r.score, reverse=True)
    elif args.score_mode == "stripe":
        stripe_results.sort(key=lambda r: r.score, reverse=True)
    elif args.score_mode != "logit":
        attn_results.sort(key=lambda r: r.score, reverse=True)
    else:
        results.sort(key=lambda r: r.logit_lb, reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    active_count = len(stripe_positions) if args.score_mode in {"stripe", "circuit", "circuit_copy"} and stripe_positions else len(active_positions)
    with args.output.open("w", encoding="ascii") as f:
        f.write("Induction discovery (approximate ranking)\n")
        f.write(f"model={args.model}\n")
        f.write(f"score_mode={args.score_mode}\n")
        f.write(f"use_activations={args.use_activations}\n")
        if args.use_activations:
            f.write(f"hf_model={args.hf_model} device={args.device}\n")
        f.write(f"tokens={len(unique_tokens)} active={active_count}\n")
        if args.score_mode in {"stripe", "circuit", "circuit_copy"}:
            f.write(f"stripe_period={args.stripe_period}\n")
        f.write(
            f"min-eps={args.min_eps} min-margin={args.min_margin} "
            f"min-logit-lb={args.min_logit_lb} min-score={args.min_score} "
            f"min-copy={args.min_copy}\n"
        )
        if args.score_mode == "circuit_copy":
            for rank, result in enumerate(circuit_copy_results[: args.top], start=1):
                f.write(f"{rank:02d} {format_circuit_copy_result(result)}\n")
        elif args.score_mode == "circuit":
            for rank, result in enumerate(circuit_results[: args.top], start=1):
                f.write(f"{rank:02d} {format_circuit_result(result)}\n")
        elif args.score_mode == "stripe":
            for rank, result in enumerate(stripe_results[: args.top], start=1):
                f.write(f"{rank:02d} {format_stripe_result(result)}\n")
        elif args.score_mode != "logit":
            for rank, result in enumerate(attn_results[: args.top], start=1):
                f.write(f"{rank:02d} {format_attn_result(result)}\n")
        else:
            for rank, result in enumerate(results[: args.top], start=1):
                f.write(f"{rank:02d} {format_result(result)}\n")

    print(f"Wrote report to {args.output}")
    if args.score_mode == "circuit_copy":
        for rank, result in enumerate(circuit_copy_results[: args.top], start=1):
            print(f"{rank:02d} {format_circuit_copy_result(result)}")
    elif args.score_mode == "circuit":
        for rank, result in enumerate(circuit_results[: args.top], start=1):
            print(f"{rank:02d} {format_circuit_result(result)}")
    elif args.score_mode == "stripe":
        for rank, result in enumerate(stripe_results[: args.top], start=1):
            print(f"{rank:02d} {format_stripe_result(result)}")
    elif args.score_mode != "logit":
        for rank, result in enumerate(attn_results[: args.top], start=1):
            print(f"{rank:02d} {format_attn_result(result)}")
    else:
        for rank, result in enumerate(results[: args.top], start=1):
            print(f"{rank:02d} {format_result(result)}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(args.model),
            "tokens": len(unique_tokens),
            "active": active_count,
            "score_mode": args.score_mode,
            "min_eps": args.min_eps,
            "min_margin": args.min_margin,
            "min_logit_lb": args.min_logit_lb,
            "min_score": args.min_score,
            "min_copy": args.min_copy,
            "use_activations": args.use_activations,
            "hf_model": args.hf_model if args.use_activations else None,
            "device": args.device if args.use_activations else None,
            "stripe_period": args.stripe_period if args.score_mode == "stripe" else None,
        }
        if args.score_mode == "circuit_copy":
            payload["results"] = [
                {
                    "rank": rank,
                    "prev_layer": r.prev_layer + 1,
                    "prev_head": r.prev_head + 1,
                    "induction_layer": r.induction_layer + 1,
                    "induction_head": r.induction_head + 1,
                    "score": r.score,
                    "prev_mean": r.prev_mean,
                    "prev_median": r.prev_median,
                    "prev_top1_frac": r.prev_top1_frac,
                    "stripe_mean": r.stripe_mean,
                    "stripe_median": r.stripe_median,
                    "stripe_top1_frac": r.stripe_top1_frac,
                    "copy_mean": r.copy_mean,
                    "copy_weighted_mean": r.copy_weighted_mean,
                }
                for rank, r in enumerate(circuit_copy_results[: args.top], start=1)
            ]
        elif args.score_mode == "circuit":
            payload["results"] = [
                {
                    "rank": rank,
                    "prev_layer": r.prev_layer + 1,
                    "prev_head": r.prev_head + 1,
                    "induction_layer": r.induction_layer + 1,
                    "induction_head": r.induction_head + 1,
                    "score": r.score,
                    "prev_mean": r.prev_mean,
                    "prev_median": r.prev_median,
                    "prev_top1_frac": r.prev_top1_frac,
                    "stripe_mean": r.stripe_mean,
                    "stripe_median": r.stripe_median,
                    "stripe_top1_frac": r.stripe_top1_frac,
                }
                for rank, r in enumerate(circuit_results[: args.top], start=1)
            ]
        elif args.score_mode == "stripe":
            payload["results"] = [
                {
                    "rank": rank,
                    "layer": r.layer + 1,
                    "head": r.head + 1,
                    "score": r.score,
                    "stripe_mean": r.stripe_mean,
                    "stripe_median": r.stripe_median,
                    "stripe_top1_frac": r.stripe_top1_frac,
                    "eps": r.eps,
                    "margin": r.margin,
                    "active": r.active,
                }
                for rank, r in enumerate(stripe_results[: args.top], start=1)
            ]
        elif args.score_mode != "logit":
            payload["results"] = [
                {
                    "rank": rank,
                    "layer": r.layer + 1,
                    "head": r.head + 1,
                    "score": r.score,
                    "prev_mean": r.prev_mean,
                    "prev_median": r.prev_median,
                    "prev_top1_frac": r.prev_top1_frac,
                    "copy_mean": r.copy_mean,
                    "copy_weighted_mean": r.copy_weighted_mean,
                    "eps": r.eps,
                    "margin": r.margin,
                    "active": r.active,
                }
                for rank, r in enumerate(attn_results[: args.top], start=1)
            ]
        else:
            payload["results"] = [
                {
                    "rank": rank,
                    "layer": r.layer + 1,
                    "head": r.head + 1,
                    "target": r.target,
                    "negative": r.negative,
                    "logit_lb": r.logit_lb,
                    "eps": r.eps,
                    "margin": r.margin,
                    "min_prev": r.min_prev,
                    "value_range": r.value_range,
                    "prev_mean": r.prev_mean,
                    "prev_median": r.prev_median,
                    "prev_top1_frac": r.prev_top1_frac,
                    "active": r.active,
                }
                for rank, r in enumerate(results[: args.top], start=1)
            ]
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="ascii")

    if args.verify_top > 0 and results:
        nfp_cmd = resolve_nfp_cmd(args.nfp_bin)
        print("\nVerifying top candidates with Lean checker:")
        for result in results[: args.verify_top]:
            cmd = nfp_cmd + [
                "induction",
                "certify_head_model_nonvacuous",
                "--model",
                str(args.model),
                "--layer",
                str(result.layer),
                "--head",
                str(result.head),
                "--direction-target",
                str(result.target),
                "--direction-negative",
                str(result.negative),
            ]
            if args.period is not None:
                cmd += ["--period", str(args.period)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            status = "ok" if proc.returncode == 0 else "fail"
            stdout = proc.stdout.strip().replace("\n", " ")
            stderr = proc.stderr.strip().replace("\n", " ")
            print(f"{status} {result.layer}/{result.head} tgt={result.target} neg={result.negative}")
            if stdout:
                print(f"  out: {stdout}")
            if stderr:
                print(f"  err: {stderr}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
