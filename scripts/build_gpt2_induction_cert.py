#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build an induction-head certificate for a GPT-2-small induction head.

This script is untrusted and uses floating-point arithmetic to produce a
rational induction-head certificate compatible with `nfp induction verify`.
Active induction positions are recorded as `active <q>` lines in the output.
All sequence indices in the certificate are 0-based (file-format convention).

The repeated-pattern inputs match the standard induction-head diagnostic setup
from the literature (pattern repeated twice).

Usage:
  python scripts/build_gpt2_induction_cert.py --output reports/gpt2_induction.cert \
    --layer 0 --head 5 --seq 32 --pattern-length 16 \
    --random-pattern --seed 0 \
    --values-out reports/gpt2_induction.values --value-dim 0 \
    --active-eps-max 0.2 --min-margin 0

Optionally, provide a logit-diff direction:
  --direction-target <tok_id> --direction-negative <tok_id>

Or ask the script to search for a direction (untrusted):
  --search-direction --direction-vocab-min 1000 --direction-vocab-max 2000
  --direction-report-out reports/direction_report.txt --direction-topk 10

Optional token dump (for Lean-side prev/active verification):
  --tokens-out reports/gpt2_induction.tokens

Optional token input (bypass random pattern):
  --tokens-in reports/gpt2_induction.tokens

Note: active positions are filtered by --active-eps-max and --min-margin. If
none qualify, the script exits with an error.
Layer/head indices are 0-based in the CLI to match the literature.
Direction token IDs use the model's raw tokenizer indexing.

To emit an induction-aligned certificate, pass:
  --kind induction-aligned
which adds a `period` entry (set to `--pattern-length`).

Induction-aligned certificates also include `copy-logit` entries derived from
the head's direct logit contribution (mean-subtracted, ReLU).
"""

import argparse
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import GPT2Model
except ImportError:
    raise SystemExit(
        "Missing dependencies. Install with: uv add transformers torch"
    )


def rat_from_float(x: float, decimals: int) -> Fraction:
    scale = 10 ** decimals
    return Fraction(int(round(x * scale)), scale)


def rat_from_float_floor(x: float, decimals: int) -> Fraction:
    scale = 10 ** decimals
    return Fraction(int(math.floor(x * scale)), scale)


def rat_from_float_exact(x: float) -> Fraction:
    return Fraction.from_float(float(x))


def rat_to_str(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def build_tokens(seq: int, pattern_len: int, random_pattern: bool, seed: int) -> np.ndarray:
    if random_pattern:
        rng = np.random.default_rng(seed)
        pattern = rng.integers(1000, 30000, size=pattern_len, endpoint=False)
    else:
        pattern = np.arange(pattern_len)
    repeats = (seq // pattern_len) + 1
    return np.tile(pattern, repeats)[:seq]


def build_prev(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prev = np.zeros_like(tokens)
    active = np.zeros_like(tokens, dtype=bool)
    last_seen = {}
    for idx, tok in enumerate(tokens):
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


def compute_scores_weights(model, input_ids, layer: int, head: int, device: str):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer]
        block = model.h[layer]
        x = block.ln_1(hidden_states)
        qkv = block.attn.c_attn(x)
        n_head = model.config.n_head
        head_dim = qkv.shape[-1] // (3 * n_head)
        q, k, _v = qkv.split(n_head * head_dim, dim=2)
        q = q.view(1, -1, n_head, head_dim).transpose(1, 2)
        k = k.view(1, -1, n_head, head_dim).transpose(1, 2)
        v = _v.view(1, -1, n_head, head_dim).transpose(1, 2)
        qh = q[:, head]
        kh = k[:, head]
        vh = v[:, head]
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(head_dim)
        seq = scores.shape[-1]
        mask = torch.triu(torch.ones(seq, seq, device=device), diagonal=1).bool()
        scores = scores.masked_fill(mask, -10000.0)
        weights = torch.softmax(scores, dim=-1)
    return (scores.squeeze(0).cpu().numpy(),
            weights.squeeze(0).cpu().numpy(),
            vh.squeeze(0).cpu().numpy())


def extract_model_slice(model, input_ids, layer: int, head: int):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer]
        block = model.h[layer]
        x = block.ln_1(hidden_states).squeeze(0).cpu().numpy()
        w = block.attn.c_attn.weight.detach().cpu().numpy()
        b = block.attn.c_attn.bias.detach().cpu().numpy()
    d_model = w.shape[0]
    n_head = model.config.n_head
    head_dim = d_model // n_head
    start = head * head_dim
    end = (head + 1) * head_dim
    wq = w[:, start:end]
    wk = w[:, d_model + start: d_model + end]
    bq = b[start:end]
    bk = b[d_model + start: d_model + end]
    scale = 1.0 / math.sqrt(head_dim)
    score_mask = -10000.0
    return {
        "d_model": int(d_model),
        "head_dim": int(head_dim),
        "layer": int(layer),
        "head": int(head),
        "scale": float(scale),
        "mask": float(score_mask),
        "resid": x,
        "wq": wq,
        "wk": wk,
        "bq": bq,
        "bk": bk,
    }


def compute_scores_from_model_slice(model_slice: dict) -> list[list[Fraction]]:
    d_model = model_slice["d_model"]
    head_dim = model_slice["head_dim"]
    scale = model_slice["scale"]
    mask = model_slice["mask"]
    resid = model_slice["resid"]
    wq = model_slice["wq"]
    wk = model_slice["wk"]
    bq = model_slice["bq"]
    bk = model_slice["bk"]
    seq = len(resid)

    q_vecs = []
    k_vecs = []
    for q in range(seq):
        q_row = []
        k_row = []
        for j in range(head_dim):
            q_sum = sum(resid[q][i] * wq[i][j] for i in range(d_model)) + bq[j]
            k_sum = sum(resid[q][i] * wk[i][j] for i in range(d_model)) + bk[j]
            q_row.append(q_sum)
            k_row.append(k_sum)
        q_vecs.append(q_row)
        k_vecs.append(k_row)

    scores = []
    for q in range(seq):
        row = []
        for k in range(seq):
            if k > q:
                row.append(mask)
            else:
                dot = sum(q_vecs[q][j] * k_vecs[k][j] for j in range(head_dim))
                row.append(scale * dot)
        scores.append(row)
    return scores


def compute_copy_logits(model, input_ids, layer: int, head: int,
                        weights: np.ndarray, values: np.ndarray, device: str) -> np.ndarray:
    head_dim = model.config.n_embd // model.config.n_head
    start, end = head * head_dim, (head + 1) * head_dim
    weights_t = torch.tensor(weights, device=device, dtype=torch.float32)
    values_t = torch.tensor(values, device=device, dtype=torch.float32)
    head_out = torch.matmul(weights_t, values_t)
    w_o = model.h[layer].attn.c_proj.weight.to(device)
    w_o_head = w_o[:, start:end]
    head_resid = torch.matmul(head_out, w_o_head.T)
    wte = model.wte.weight.to(device)
    logits = torch.matmul(head_resid, wte.T)
    logits = logits - logits.mean(dim=-1, keepdim=True)
    logits = torch.relu(logits)
    token_ids = input_ids.squeeze(0)
    copy_logits = logits[:, token_ids]
    return copy_logits.detach().cpu().numpy()


def compute_tl_mul_score(
    weights: np.ndarray,
    tokens: np.ndarray,
    exclude_bos: bool,
    exclude_current_token: bool,
) -> float:
    seq = len(tokens)
    numerator = 0.0
    denominator = 0.0
    for q in range(seq):
        for k in range(seq):
            w = float(weights[q][k])
            if exclude_bos and k == 0:
                w = 0.0
            if exclude_current_token and k == q:
                w = 0.0
            denominator += w
            if k > 0 and (k - 1) < q and tokens[k - 1] == tokens[q]:
                numerator += w
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def write_scores(path: Path, seq: int, prev: np.ndarray, scores, weights, eps=None, margin=None,
                 active=None, kind: str = "onehot-approx", period: int | None = None):
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {seq}\n")
        f.write(f"kind {kind}\n")
        if kind == "induction-aligned":
            if period is None:
                raise ValueError("period is required for induction-aligned certificates")
            f.write(f"period {period}\n")
        if eps is not None:
            f.write(f"eps {rat_to_str(eps)}\n")
        if margin is not None:
            f.write(f"margin {rat_to_str(margin)}\n")
        if active is not None:
            for q in active:
                f.write(f"active {q}\n")
        for q, k in enumerate(prev.tolist()):
            f.write(f"prev {q} {k}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"score {q} {k} {rat_to_str(scores[q][k])}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"weight {q} {k} {rat_to_str(weights[q][k])}\n")

def write_induction_cert(path: Path, seq: int, prev: np.ndarray, scores, weights,
                         eps, margin, active, eps_at, weight_bound_at,
                         vals, direction_target=None, direction_negative=None,
                         kind: str = "onehot-approx", period: int | None = None,
                         copy_logits=None,
                         tl_score_lb: Fraction | None = None,
                         tl_exclude_bos: bool = False,
                         tl_exclude_current_token: bool = False,
                         model_slice=None) -> None:
    lo = min(vals)
    hi = max(vals)
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {seq}\n")
        f.write(f"kind {kind}\n")
        if kind == "induction-aligned":
            if period is None:
                raise ValueError("period is required for induction-aligned certificates")
            f.write(f"period {period}\n")
        if model_slice is not None:
            f.write(f"model-d-model {model_slice['d_model']}\n")
            f.write(f"model-head-dim {model_slice['head_dim']}\n")
            f.write(f"model-layer {model_slice['layer']}\n")
            f.write(f"model-head {model_slice['head']}\n")
            f.write(f"model-score-scale {rat_to_str(model_slice['scale'])}\n")
            f.write(f"model-score-mask {rat_to_str(model_slice['mask'])}\n")
            resid = model_slice["resid"]
            wq = model_slice["wq"]
            wk = model_slice["wk"]
            bq = model_slice["bq"]
            bk = model_slice["bk"]
            d_model = model_slice["d_model"]
            head_dim = model_slice["head_dim"]
            for q in range(seq):
                for i in range(d_model):
                    f.write(f"model-resid {q} {i} {rat_to_str(resid[q][i])}\n")
            for i in range(d_model):
                for j in range(head_dim):
                    f.write(f"model-wq {i} {j} {rat_to_str(wq[i][j])}\n")
            for i in range(d_model):
                for j in range(head_dim):
                    f.write(f"model-wk {i} {j} {rat_to_str(wk[i][j])}\n")
            for j in range(head_dim):
                f.write(f"model-bq {j} {rat_to_str(bq[j])}\n")
            for j in range(head_dim):
                f.write(f"model-bk {j} {rat_to_str(bk[j])}\n")
        if tl_score_lb is not None:
            f.write(f"tl-exclude-bos {str(tl_exclude_bos).lower()}\n")
            f.write(f"tl-exclude-current-token {str(tl_exclude_current_token).lower()}\n")
            f.write(f"tl-score-lb {rat_to_str(tl_score_lb)}\n")
        if direction_target is not None and direction_negative is not None:
            f.write(f"direction-target {direction_target}\n")
            f.write(f"direction-negative {direction_negative}\n")
        f.write(f"eps {rat_to_str(eps)}\n")
        f.write(f"margin {rat_to_str(margin)}\n")
        if active is not None:
            for q in active:
                f.write(f"active {q}\n")
        for q, k in enumerate(prev.tolist()):
            f.write(f"prev {q} {k}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"score {q} {k} {rat_to_str(scores[q][k])}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"weight {q} {k} {rat_to_str(weights[q][k])}\n")
        if copy_logits is not None:
            for q in range(seq):
                for k in range(seq):
                    f.write(f"copy-logit {q} {k} {rat_to_str(copy_logits[q][k])}\n")
        for q in range(seq):
            f.write(f"eps-at {q} {rat_to_str(eps_at[q])}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"weight-bound {q} {k} {rat_to_str(weight_bound_at[q][k])}\n")
        f.write(f"lo {rat_to_str(lo)}\n")
        f.write(f"hi {rat_to_str(hi)}\n")
        for k, val in enumerate(vals):
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
        f.write("kind onehot-approx\n")
        if direction_target is not None and direction_negative is not None:
            f.write(f"direction-target {direction_target}\n")
            f.write(f"direction-negative {direction_negative}\n")
        f.write(f"lo {rat_to_str(lo)}\n")
        f.write(f"hi {rat_to_str(hi)}\n")
        for k, val in enumerate(vals_rat):
            f.write(f"val {k} {rat_to_str(val)}\n")


def write_tokens(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {len(tokens)}\n")
        for idx, tok in enumerate(tokens.tolist()):
            f.write(f"token {idx} {tok}\n")


def read_tokens(path: Path) -> np.ndarray:
    seq = None
    tokens: dict[int, int] = {}
    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "seq" and len(parts) == 2:
            if seq is not None:
                raise ValueError("duplicate seq entry in tokens file")
            seq = int(parts[1])
            continue
        if parts[0] == "token" and len(parts) == 3:
            idx = int(parts[1])
            tok = int(parts[2])
            if idx in tokens:
                raise ValueError(f"duplicate token entry for index {idx}")
            tokens[idx] = tok
            continue
        raise ValueError(f"unrecognized tokens line: {line}")

    if seq is None:
        raise ValueError("missing seq entry in tokens file")
    if seq <= 0:
        raise ValueError("seq must be positive in tokens file")
    if len(tokens) != seq:
        raise ValueError("tokens file missing entries (expected seq tokens)")
    arr = np.zeros(seq, dtype=np.int64)
    for idx in range(seq):
        if idx not in tokens:
            raise ValueError(f"missing token entry for index {idx}")
        arr[idx] = tokens[idx]
    return arr

def search_direction(
    model,
    values: np.ndarray,
    layer: int,
    head: int,
    prev: np.ndarray,
    active_positions: list[int],
    eps_at: list[float],
    vocab_min: int,
    vocab_max: int,
    max_candidates: int | None,
    topk: int,
    seed: int,
) -> tuple[int, int, float, list[tuple[float, int, int]]]:
    import heapq

    if vocab_min < 0 or vocab_max <= vocab_min:
        raise SystemExit("direction vocab range must satisfy 0 <= min < max")
    cand_ids = list(range(vocab_min, vocab_max))
    if max_candidates is not None and max_candidates > 0 and len(cand_ids) > max_candidates:
        rng = np.random.default_rng(seed)
        cand_ids = rng.choice(cand_ids, size=max_candidates, replace=False).tolist()

    wte = model.wte.weight.detach().cpu().numpy()
    if max(cand_ids) >= wte.shape[0]:
        raise SystemExit("direction vocab range exceeds model vocab size")
    head_dim = model.config.n_embd // model.config.n_head
    start, end = head * head_dim, (head + 1) * head_dim
    w_o = model.h[layer].attn.c_proj.weight.detach().cpu().numpy()[:, start:end]

    proj = wte[cand_ids] @ w_o  # (m, head_dim)
    vals_mat = values @ proj.T  # (seq, m)
    active_prev = [(q, int(prev[q])) for q in active_positions]
    best_lb = None
    best_pair = None
    topk_entries: list[tuple[float, int, int]] = []

    for i in range(vals_mat.shape[1]):
        vals_i = vals_mat[:, i]
        for j in range(vals_mat.shape[1]):
            if i == j:
                continue
            vals = vals_i - vals_mat[:, j]
            lo = float(vals.min())
            hi = float(vals.max())
            gap = hi - lo
            lb = None
            for q, pq in active_prev:
                cand = float(vals[pq]) - eps_at[q] * gap
                if lb is None or cand < lb:
                    lb = cand
            if lb is None:
                continue
            if best_lb is None or lb > best_lb:
                best_lb = lb
                best_pair = (cand_ids[i], cand_ids[j])
            if topk > 0:
                # Preserve the best few candidates for reporting.
                if len(topk_entries) < topk:
                    heapq.heappush(topk_entries, (lb, cand_ids[i], cand_ids[j]))
                else:
                    if lb > topk_entries[0][0]:
                        heapq.heapreplace(topk_entries, (lb, cand_ids[i], cand_ids[j]))

    if best_pair is None or best_lb is None:
        raise SystemExit("No direction candidates found")
    topk_sorted = sorted(topk_entries, key=lambda x: x[0], reverse=True)
    return best_pair[0], best_pair[1], best_lb, topk_sorted


def write_direction_report(
    path: Path,
    entries: list[tuple[float, int, int]],
    vocab_min: int,
    vocab_max: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("direction_report\n")
        f.write(f"vocab_min={vocab_min} vocab_max={vocab_max} seed={seed}\n")
        f.write("rank\tlb\ttarget\tnegative\n")
        for rank, (lb, target, negative) in enumerate(entries, start=1):
            f.write(f"{rank}\t{lb:.6f}\t{target}\t{negative}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to write certificate")
    parser.add_argument("--scores-out", help="Optional path for raw scores/weights dump")
    parser.add_argument("--layer", type=int, default=0,
                        help="Transformer layer index (0-based)")
    parser.add_argument("--head", type=int, default=0,
                        help="Attention head index (0-based)")
    parser.add_argument("--seq", type=int, default=32, help="Sequence length")
    parser.add_argument("--pattern-length", type=int, default=16, help="Pattern length")
    parser.add_argument("--random-pattern", action="store_true", help="Use random token pattern")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random pattern")
    parser.add_argument("--kind", default="onehot-approx",
                        choices=["onehot-approx", "induction-aligned"],
                        help="Certificate kind (default: onehot-approx).")
    parser.add_argument("--decimals", type=int, default=6, help="Decimal rounding for rationals")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--values-out", help="Optional path for a value-range certificate")
    parser.add_argument("--value-dim", type=int, default=0,
                        help="Value dimension index for the value-range certificate")
    parser.add_argument("--tokens-in", type=Path, help="Optional path to load the token list")
    parser.add_argument("--tokens-out", help="Optional path to write the token list")
    parser.add_argument("--active-eps-max", default="1/2",
                        help="Maximum eps to include an active position (default: 1/2).")
    parser.add_argument("--min-margin", default="0",
                        help="Minimum score gap required for an active position (default: 0).")
    parser.add_argument("--direction-target", type=int,
                        help="Target token id for logit-diff direction (optional)")
    parser.add_argument("--direction-negative", type=int,
                        help="Negative token id for logit-diff direction (optional)")
    parser.add_argument("--search-direction", action="store_true",
                        help="Search for a direction within the vocab range (untrusted).")
    parser.add_argument("--direction-vocab-min", type=int, default=1000,
                        help="Minimum vocab id for direction search (inclusive).")
    parser.add_argument("--direction-vocab-max", type=int, default=2000,
                        help="Maximum vocab id for direction search (exclusive).")
    parser.add_argument("--direction-max-candidates", type=int, default=0,
                        help="Limit number of direction candidates (0 = all in range).")
    parser.add_argument("--direction-min-lb", default="0",
                        help="Minimum logit-diff lower bound to accept (default: 0).")
    parser.add_argument("--direction-report-out", type=Path,
                        help="Optional path to write a ranked direction report.")
    parser.add_argument("--direction-topk", type=int, default=10,
                        help="How many top directions to report (default: 10).")
    parser.add_argument("--tl-score", action="store_true",
                        help="Include TL mul score lower bound in the certificate.")
    parser.add_argument("--tl-exclude-bos", action="store_true",
                        help="Match TL exclude_bos when computing TL score.")
    parser.add_argument("--tl-exclude-current-token", action="store_true",
                        help="Match TL exclude_current_token when computing TL score.")
    parser.add_argument("--tl-score-decimals", type=int, default=None,
                        help="Decimal rounding for TL score LB (defaults to --decimals).")
    parser.add_argument("--emit-model-slice", action="store_true",
                        help="Embed model slice used to compute attention scores.")
    args = parser.parse_args()

    tokens = None
    if args.tokens_in is not None:
        try:
            tokens = read_tokens(args.tokens_in)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.seq != len(tokens):
            print("warning: overriding --seq to match tokens-in length")
            args.seq = len(tokens)
        if args.random_pattern:
            print("warning: tokens-in provided; ignoring --random-pattern and --seed")
    else:
        if args.seq <= 0:
            raise SystemExit("seq must be positive")
        tokens = build_tokens(args.seq, args.pattern_length, args.random_pattern, args.seed)

    if args.kind == "induction-aligned":
        if args.pattern_length <= 0:
            raise SystemExit("pattern-length must be positive for induction-aligned")
        if args.pattern_length >= args.seq:
            raise SystemExit("pattern-length must be less than seq for induction-aligned")
    prev, active_mask = build_prev(tokens)
    candidate_positions = [int(i) for i, flag in enumerate(active_mask) if flag]

    if args.layer < 0:
        raise SystemExit("layer must be >= 0")
    if args.head < 0:
        raise SystemExit("head must be >= 0")

    model = GPT2Model.from_pretrained(args.model)
    layer = args.layer
    head = args.head
    if layer >= model.config.n_layer:
        raise SystemExit(f"layer must be in [0, {model.config.n_layer - 1}]")
    if head >= model.config.n_head:
        raise SystemExit(f"head must be in [0, {model.config.n_head - 1}]")
    model.to(args.device)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=args.device).unsqueeze(0)

    scores, weights, values = compute_scores_weights(model, input_ids, layer, head, args.device)
    copy_logits = None
    if args.kind == "induction-aligned":
        copy_logits = compute_copy_logits(
            model, input_ids, layer, head, weights, values, args.device
        )

    model_slice = None
    if args.emit_model_slice:
        raw_slice = extract_model_slice(model, input_ids, layer, head)
        resid = [
            [rat_from_float_exact(float(raw_slice["resid"][q, i]))
             for i in range(raw_slice["d_model"])]
            for q in range(args.seq)
        ]
        wq = [
            [rat_from_float_exact(float(raw_slice["wq"][i, j]))
             for j in range(raw_slice["head_dim"])]
            for i in range(raw_slice["d_model"])
        ]
        wk = [
            [rat_from_float_exact(float(raw_slice["wk"][i, j]))
             for j in range(raw_slice["head_dim"])]
            for i in range(raw_slice["d_model"])
        ]
        bq = [rat_from_float_exact(float(raw_slice["bq"][j]))
              for j in range(raw_slice["head_dim"])]
        bk = [rat_from_float_exact(float(raw_slice["bk"][j]))
              for j in range(raw_slice["head_dim"])]
        model_slice = {
            "d_model": raw_slice["d_model"],
            "head_dim": raw_slice["head_dim"],
            "layer": raw_slice["layer"],
            "head": raw_slice["head"],
            "scale": rat_from_float_exact(raw_slice["scale"]),
            "mask": rat_from_float_exact(raw_slice["mask"]),
            "resid": resid,
            "wq": wq,
            "wk": wk,
            "bq": bq,
            "bk": bk,
        }
        scores_rat = compute_scores_from_model_slice(model_slice)
    else:
        scores_rat = [[rat_from_float(float(scores[q, k]), args.decimals) for k in range(args.seq)]
                      for q in range(args.seq)]
    weights_rat = [[rat_from_float(float(weights[q, k]), args.decimals) for k in range(args.seq)]
                   for q in range(args.seq)]
    copy_logits_rat = None
    if copy_logits is not None:
        copy_logits_rat = [
            [rat_from_float(float(copy_logits[q, k]), args.decimals) for k in range(args.seq)]
            for q in range(args.seq)
        ]

    for q in range(args.seq):
        total = sum(weights_rat[q], Fraction(0))
        if total == 0:
            raise SystemExit(f"zero weight sum at q={q}")
        weights_rat[q] = [w / total for w in weights_rat[q]]

    eps = Fraction(0)
    margin = None
    eps_by_q: dict[int, Fraction] = {}
    margin_by_q: dict[int, Fraction] = {}
    for q in range(args.seq):
        prev_q = prev[q]
        prev_w = weights_rat[q][prev_q]
        if args.seq == 1:
            max_other = Fraction(0)
        else:
            max_other = max(weights_rat[q][k] for k in range(args.seq) if k != prev_q)
        deficit = Fraction(1) - prev_w
        eps_by_q[q] = max(max_other, deficit)

        diffs = [scores_rat[q][prev_q] - scores_rat[q][k]
                 for k in range(args.seq) if k != prev_q]
        margin_by_q[q] = min(diffs) if diffs else Fraction(0)

    eps_threshold = Fraction(args.active_eps_max)
    min_margin = Fraction(args.min_margin)
    if args.kind == "induction-aligned":
        active_positions = [q for q in range(args.seq) if q >= args.pattern_length]
    else:
        active_positions = [
            q for q in candidate_positions
            if eps_by_q[q] <= eps_threshold and margin_by_q[q] >= min_margin
        ]
        if not active_positions:
            raise SystemExit(
                "No active positions satisfy active-eps-max/min-margin. "
                "Try a different head/layer, random-pattern/seed, or relax the thresholds."
            )
    if active_positions:
        eps = max(eps_by_q[q] for q in active_positions)
        margin = min((margin_by_q[q] for q in active_positions), default=Fraction(0))
    else:
        margin = Fraction(0)

    if candidate_positions:
        print(f"Active positions: {len(active_positions)}/{len(candidate_positions)}")

    eps_at = []
    for q in range(args.seq):
        prev_q = prev[q]
        if args.seq == 1:
            max_other = Fraction(0)
        else:
            max_other = max(weights_rat[q][k] for k in range(args.seq) if k != prev_q)
        deficit = Fraction(1) - weights_rat[q][prev_q]
        eps_at.append(max(max_other, deficit))
    weight_bound_at = weights_rat

    if args.search_direction and (args.direction_target is not None or args.direction_negative is not None):
        raise SystemExit("search-direction is mutually exclusive with explicit direction tokens")
    if args.direction_report_out is not None and not args.search_direction:
        raise SystemExit("direction-report-out requires --search-direction")

    direction_target = None
    direction_negative = None
    if args.search_direction:
        eps_at_float = [float(eps_at[q]) for q in range(args.seq)]
        max_candidates = args.direction_max_candidates
        if max_candidates == 0:
            max_candidates = None
        direction_target, direction_negative, best_lb, topk_entries = search_direction(
            model=model,
            values=values,
            layer=layer,
            head=head,
            prev=prev,
            active_positions=active_positions,
            eps_at=eps_at_float,
            vocab_min=args.direction_vocab_min,
            vocab_max=args.direction_vocab_max,
            max_candidates=max_candidates,
            topk=args.direction_topk,
            seed=args.seed,
        )
        if args.direction_report_out is not None:
            write_direction_report(
                args.direction_report_out,
                topk_entries,
                args.direction_vocab_min,
                args.direction_vocab_max,
                args.seed,
            )
        try:
            min_lb = float(Fraction(args.direction_min_lb))
        except (ValueError, ZeroDivisionError) as exc:
            raise SystemExit("direction-min-lb must be a rational literal") from exc
        if best_lb < min_lb:
            raise SystemExit(
                f"Best direction lower bound {best_lb:.6f} below minimum {min_lb:.6f}."
            )
        print(
            f"Selected direction: target={direction_target} negative={direction_negative} "
            f"(estimated LB {best_lb:.6f})"
        )

    if (args.direction_target is None) != (args.direction_negative is None):
        raise SystemExit("direction-target and direction-negative must be provided together")
    if args.direction_target is not None:
        direction_target = args.direction_target
        direction_negative = args.direction_negative

    if direction_target is not None:
        wte = model.wte.weight.detach().cpu().numpy()
        if direction_target < 0 or direction_target >= wte.shape[0]:
            raise SystemExit("direction-target out of vocab range")
        if direction_negative < 0 or direction_negative >= wte.shape[0]:
            raise SystemExit("direction-negative out of vocab range")
        direction = wte[direction_target] - wte[direction_negative]
        head_dim = model.config.n_embd // model.config.n_head
        start, end = head * head_dim, (head + 1) * head_dim
        w_o = model.h[layer].attn.c_proj.weight.detach().cpu().numpy()[:, start:end]
        dir_head = w_o.T @ direction
        vals = values @ dir_head
    else:
        if args.value_dim < 0 or args.value_dim >= values.shape[1]:
            raise SystemExit(f"value-dim must be in [0, {values.shape[1] - 1}]")
        vals = values[:, args.value_dim]

    vals_rat = [rat_from_float(float(vals[k]), args.decimals) for k in range(args.seq)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    period = args.pattern_length if args.kind == "induction-aligned" else None
    tl_score_lb = None
    if args.tl_score:
        tl_score = compute_tl_mul_score(
            weights,
            tokens,
            exclude_bos=args.tl_exclude_bos,
            exclude_current_token=args.tl_exclude_current_token,
        )
        tl_decimals = args.decimals if args.tl_score_decimals is None else args.tl_score_decimals
        tl_score_lb = rat_from_float_floor(tl_score, tl_decimals)

    write_induction_cert(
        output_path,
        args.seq,
        prev,
        scores_rat,
        weights_rat,
        eps,
        margin,
        active_positions,
        eps_at,
        weight_bound_at,
        vals_rat,
        direction_target=direction_target,
        direction_negative=direction_negative,
        kind=args.kind,
        period=period,
        copy_logits=copy_logits_rat,
        tl_score_lb=tl_score_lb,
        tl_exclude_bos=args.tl_exclude_bos,
        tl_exclude_current_token=args.tl_exclude_current_token,
        model_slice=model_slice,
    )

    if args.scores_out:
        scores_path = Path(args.scores_out)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        write_scores(scores_path, args.seq, prev, scores_rat, weights_rat,
                     active=active_positions, kind=args.kind, period=period)

    if args.values_out:
        values_path = Path(args.values_out)
        values_path.parent.mkdir(parents=True, exist_ok=True)
        write_value_range(values_path, args.seq, vals, args.decimals,
                          direction_target=direction_target,
                          direction_negative=direction_negative)
    if args.tokens_out:
        tokens_path = Path(args.tokens_out)
        write_tokens(tokens_path, tokens)

    print(f"Wrote certificate to {output_path}")
    if args.scores_out:
        print(f"Wrote scores dump to {scores_path}")
    if args.values_out:
        print(f"Wrote value-range certificate to {values_path}")
    if args.tokens_out:
        print(f"Wrote token list to {tokens_path}")


if __name__ == "__main__":
    main()
