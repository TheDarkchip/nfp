#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Generate a "Strong Induction" dataset for NFP verification.
Creates a sequence of random common words repeated multiple times.
"""

import sys
from pathlib import Path
import numpy as np
import torch

try:
    from transformers import GPT2Model
except ImportError:
    print("Error: transformers library not installed.")
    print("Install with: uv add transformers torch")
    sys.exit(1)


def export_induction_weights(output_path: str = "models/gpt2_induction.nfpt"):
    print("Loading GPT-2 Small...")
    model = GPT2Model.from_pretrained("gpt2")
    config = model.config

    # Fixed parameters for GPT-2 Small
    seq_len = 256
    model_dim = 768
    num_heads = 12
    head_dim = 64

    # --- CRITICAL CHANGE: Better Induction Data ---
    # Instead of tokens 0-15, we use random "word" tokens (1000-30000).
    # A pattern of length 30 repeated ~8 times creates strong induction pressure.
    np.random.seed(42)
    pattern_len = 30
    # Generate 30 random token IDs
    pattern = np.random.randint(1000, 30000, size=pattern_len)
    # Repeat the pattern to fill seq_len
    repeats = (seq_len // pattern_len) + 1
    sample_tokens = np.tile(pattern, repeats)[:seq_len]

    print(f"Generated induction sequence (len={seq_len}):")
    print(f"Pattern (first 10): {sample_tokens[:10]}")
    # -----------------------------------------------

    # Compute embeddings
    wte = model.wte.weight.detach().numpy()
    wpe = model.wpe.weight.detach().numpy()
    sample_embeddings = wte[sample_tokens] + wpe[:seq_len]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {output_path}...")
    with open(output_path, "wb") as f:
        write_header(
            f,
            num_layers=config.n_layer,
            num_heads=config.n_head,
            model_dim=model_dim,
            head_dim=head_dim,
            hidden_dim=config.n_inner or 4 * model_dim,
            vocab_size=config.vocab_size,
            seq_len=seq_len,
        )

        write_i32(f, sample_tokens)
        write_f64(f, sample_embeddings)

        # Export weights
        for layer_idx in range(config.n_layer):
            block = model.h[layer_idx]

            # Attention
            c_attn = block.attn.c_attn.weight.detach().numpy()
            c_attn_bias = get_bias(block.attn.c_attn.bias, 3 * model_dim)
            c_proj = block.attn.c_proj.weight.detach().numpy()
            c_proj_bias = get_bias(block.attn.c_proj.bias, model_dim)

            W_Q_all = c_attn[:, 0:model_dim]
            W_K_all = c_attn[:, model_dim : 2 * model_dim]
            W_V_all = c_attn[:, 2 * model_dim : 3 * model_dim]
            b_Q_all = c_attn_bias[0:model_dim]
            b_K_all = c_attn_bias[model_dim : 2 * model_dim]
            b_V_all = c_attn_bias[2 * model_dim : 3 * model_dim]

            for h in range(num_heads):
                start, end = h * head_dim, (h + 1) * head_dim
                write_f64(f, W_Q_all[:, start:end])
                write_f64(f, b_Q_all[start:end])
                write_f64(f, W_K_all[:, start:end])
                write_f64(f, b_K_all[start:end])
                write_f64(f, W_V_all[:, start:end])
                write_f64(f, b_V_all[start:end])
                write_f64(f, c_proj[start:end, :])

            write_f64(f, c_proj_bias)

            # MLP
            write_f64(f, block.mlp.c_fc.weight.detach().numpy())
            write_f64(f, get_bias(block.mlp.c_fc.bias, config.n_inner or 4 * model_dim))
            write_f64(f, block.mlp.c_proj.weight.detach().numpy())
            write_f64(f, get_bias(block.mlp.c_proj.bias, model_dim))

            # LayerNorm
            write_f64(f, block.ln_1.weight.detach().numpy())
            write_f64(f, get_bias(block.ln_1.bias, model_dim))
            write_f64(f, block.ln_2.weight.detach().numpy())
            write_f64(f, get_bias(block.ln_2.bias, model_dim))

        # Unembedding
        write_f64(f, model.ln_f.weight.detach().numpy())
        write_f64(f, get_bias(model.ln_f.bias, model_dim))
        write_f64(f, wte.T)

    print("Done.")


def write_header(f, **fields: int) -> None:
    f.write(b"NFP_BINARY_V1\n")
    for key, value in fields.items():
        f.write(f"{key}={value}\n".encode("ascii"))
    f.write(b"BINARY_START\n")


def write_i32(f, data: np.ndarray) -> None:
    arr = np.ascontiguousarray(data, dtype="<i4")
    f.write(arr.tobytes(order="C"))


def write_f64(f, data: np.ndarray) -> None:
    arr = np.ascontiguousarray(data, dtype="<f8")
    f.write(arr.tobytes(order="C"))


def get_bias(param, size: int) -> np.ndarray:
    if param is None:
        return np.zeros(size, dtype=np.float64)
    return param.detach().numpy()


if __name__ == "__main__":
    export_induction_weights()
