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
    with open(output_path, "w") as f:
        # Header
        f.write("NFP_TEXT_V2\n")
        f.write(f"num_layers={config.n_layer}\n")
        f.write(f"num_heads={config.n_head}\n")
        f.write(f"model_dim={model_dim}\n")
        f.write(f"head_dim={head_dim}\n")
        f.write(f"hidden_dim={config.n_inner or 4 * model_dim}\n")
        f.write(f"vocab_size={config.vocab_size}\n")
        f.write(f"seq_len={seq_len}\n\n")

        f.write("TOKENS\n")
        write_tokens(f, sample_tokens)
        f.write("\nEMBEDDINGS\n")
        write_matrix(f, sample_embeddings)

        # Export weights
        for layer_idx in range(config.n_layer):
            block = model.h[layer_idx]
            f.write(f"LAYER {layer_idx}\n")

            # Attention
            c_attn = block.attn.c_attn.weight.detach().numpy()
            c_proj = block.attn.c_proj.weight.detach().numpy()

            W_Q_all = c_attn[:, 0:model_dim]
            W_K_all = c_attn[:, model_dim : 2 * model_dim]
            W_V_all = c_attn[:, 2 * model_dim : 3 * model_dim]

            for h in range(num_heads):
                start, end = h * head_dim, (h + 1) * head_dim
                f.write(f"HEAD {h}\n")
                f.write("W_Q\n")
                write_matrix(f, W_Q_all[:, start:end])
                f.write("W_K\n")
                write_matrix(f, W_K_all[:, start:end])
                f.write("W_V\n")
                write_matrix(f, W_V_all[:, start:end])
                f.write("W_O\n")
                write_matrix(f, c_proj[start:end, :])

            # MLP
            f.write("MLP\n")
            f.write("W_in\n")
            write_matrix(f, block.mlp.c_fc.weight.detach().numpy())
            f.write("b_in\n")
            write_vector(f, block.mlp.c_fc.bias.detach().numpy())
            f.write("W_out\n")
            write_matrix(f, block.mlp.c_proj.weight.detach().numpy())
            f.write("b_out\n")
            write_vector(f, block.mlp.c_proj.bias.detach().numpy())

            # LayerNorm
            f.write("LN1_GAMMA\n")
            write_vector(f, block.ln_1.weight.detach().numpy())
            f.write("LN1_BETA\n")
            write_vector(f, block.ln_1.bias.detach().numpy())
            f.write("LN2_GAMMA\n")
            write_vector(f, block.ln_2.weight.detach().numpy())
            f.write("LN2_BETA\n")
            write_vector(f, block.ln_2.bias.detach().numpy())

        # Unembedding
        f.write("LN_F_GAMMA\n")
        write_vector(f, model.ln_f.weight.detach().numpy())
        f.write("LN_F_BETA\n")
        write_vector(f, model.ln_f.bias.detach().numpy())
        f.write("UNEMBEDDING\n")
        write_matrix(f, wte.T)

    print("Done.")


def write_matrix(f, m):
    for row in m:
        f.write(" ".join(f"{x:.9g}" for x in row) + "\n")


def write_vector(f, v):
    f.write(" ".join(f"{x:.9g}" for x in v) + "\n")


def write_tokens(f, t):
    s = [str(x) for x in t]
    for i in range(0, len(s), 64):
        f.write(" ".join(s[i : i + 64]) + "\n")


if __name__ == "__main__":
    export_induction_weights()
