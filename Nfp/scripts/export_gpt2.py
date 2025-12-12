#!/usr/bin/env python3
"""
Export GPT-2 Small weights to NFP Text Format (.nfpt).

This script loads GPT-2 Small from HuggingFace and exports all weights needed
for circuit analysis in the NFP library.

Usage:
    uv run scripts/export_gpt2.py [output_path]

Default output: models/gpt2.nfpt
"""

import sys
from pathlib import Path
import numpy as np

try:
    from transformers import GPT2Model
except ImportError:
    print("Error: transformers library not installed.")
    print("Install with: uv add transformers torch")
    sys.exit(1)


def export_gpt2_weights(output_path: str = "models/gpt2.nfpt", seq_len: int = 256):
    """
    Export GPT-2 Small weights to .nfpt format.

    GPT-2 Small architecture:
    - 12 layers
    - 12 attention heads per layer
    - 768 model dimension
    - 64 head dimension (768 / 12)
    - 3072 MLP hidden dimension (4 * 768)
    - 50257 vocabulary size

    Args:
        output_path: Path to write the .nfpt file
        seq_len: Sequence length for analysis (default 256, smaller than GPT-2's 1024 for speed)
    """
    print("Loading GPT-2 Small from HuggingFace...")
    model = GPT2Model.from_pretrained("gpt2")
    config = model.config

    # Model dimensions
    num_layers = config.n_layer  # 12
    num_heads = config.n_head  # 12
    model_dim = config.n_embd  # 768
    head_dim = model_dim // num_heads  # 64
    hidden_dim = config.n_inner or 4 * model_dim  # 3072
    vocab_size = config.vocab_size  # 50257

    print("Model configuration:")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")
    print(f"  Model dim: {model_dim}")
    print(f"  Head dim: {head_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Sequence length (for analysis): {seq_len}")

    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting to {output_path}...")

    with open(output_path, "w") as f:
        # Header
        f.write("NFP_TEXT_V1\n")
        f.write(f"num_layers={num_layers}\n")
        f.write(f"num_heads={num_heads}\n")
        f.write(f"model_dim={model_dim}\n")
        f.write(f"head_dim={head_dim}\n")
        f.write(f"hidden_dim={hidden_dim}\n")
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"seq_len={seq_len}\n")
        f.write("\n")

        # Sample input embeddings for analysis
        # For circuit analysis, we need input embeddings for a specific sequence
        # We'll create a dummy sequence of token indices and get their embeddings
        # Shape: (seq_len, model_dim)
        wte = model.wte.weight.detach().numpy()
        wpe = model.wpe.weight.detach().numpy()  # Position embeddings

        # Create sample token indices (repeating pattern to make induction detectable)
        # Pattern: 0, 1, 2, ..., 15, 0, 1, 2, ..., 15, ...
        sample_tokens = np.array([i % 16 for i in range(seq_len)])
        # Include both token embeddings and position embeddings
        # This is what GPT-2 computes: x_0 = token_emb + pos_emb
        sample_embeddings = wte[sample_tokens] + wpe[:seq_len]  # (seq_len, model_dim)

        f.write("EMBEDDINGS\n")
        write_matrix(f, sample_embeddings)

        # Process each layer
        for layer_idx in range(num_layers):
            print(f"  Processing layer {layer_idx}...")
            block = model.h[layer_idx]

            f.write(f"LAYER {layer_idx}\n")

            # Attention weights
            # GPT-2 stores Q, K, V concatenated in c_attn.weight
            # Shape: (model_dim, 3 * model_dim) for weight
            # We need to split into Q, K, V and then split each by heads

            c_attn_weight = block.attn.c_attn.weight.detach().numpy()  # (768, 2304)

            # c_attn.weight layout: input_dim -> [Q_all_heads, K_all_heads, V_all_heads]
            # Each section is (model_dim, model_dim)
            W_Q_all = c_attn_weight[:, 0:model_dim]  # (768, 768)
            W_K_all = c_attn_weight[:, model_dim : 2 * model_dim]  # (768, 768)
            W_V_all = c_attn_weight[:, 2 * model_dim : 3 * model_dim]  # (768, 768)

            # Output projection c_proj
            c_proj_weight = block.attn.c_proj.weight.detach().numpy()  # (768, 768)

            # Split into heads
            # W_Q_all columns are organized as [head0, head1, ..., head11]
            # Each head gets head_dim columns
            for head_idx in range(num_heads):
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim

                # Extract per-head Q, K, V projections
                # W_Q: (model_dim, head_dim) - projects input to queries for this head
                W_Q = W_Q_all[:, start:end]  # (768, 64)
                W_K = W_K_all[:, start:end]  # (768, 64)
                W_V = W_V_all[:, start:end]  # (768, 64)

                # Output projection for this head
                # c_proj.weight: (768, 768), organized as [head0_rows, head1_rows, ...]
                # W_O: (head_dim, model_dim) - projects head output back to model dim
                W_O = c_proj_weight[start:end, :]  # (64, 768)

                f.write(f"HEAD {head_idx}\n")
                f.write("W_Q\n")
                write_matrix(f, W_Q)
                f.write("W_K\n")
                write_matrix(f, W_K)
                f.write("W_V\n")
                write_matrix(f, W_V)
                f.write("W_O\n")
                write_matrix(f, W_O)

            # MLP weights
            # GPT-2 MLP: x -> c_fc (expand) -> GELU -> c_proj (contract) -> out
            # c_fc: (model_dim, hidden_dim)
            # c_proj: (hidden_dim, model_dim)

            mlp_c_fc_weight = block.mlp.c_fc.weight.detach().numpy()  # (768, 3072)
            mlp_c_fc_bias = block.mlp.c_fc.bias.detach().numpy()  # (3072,)
            mlp_c_proj_weight = block.mlp.c_proj.weight.detach().numpy()  # (3072, 768)
            mlp_c_proj_bias = block.mlp.c_proj.bias.detach().numpy()  # (768,)

            # Note: GPT-2 uses Conv1D which stores weights transposed compared to Linear
            # Conv1D weight shape is (in_features, out_features)
            # Our format expects W_in: (model_dim, hidden_dim) and W_out: (hidden_dim, model_dim)

            f.write("MLP\n")
            f.write("W_in\n")
            write_matrix(f, mlp_c_fc_weight)  # (768, 3072)
            f.write("b_in\n")
            write_vector(f, mlp_c_fc_bias)  # (3072,)
            f.write("W_out\n")
            write_matrix(f, mlp_c_proj_weight)  # (3072, 768)
            f.write("b_out\n")
            write_vector(f, mlp_c_proj_bias)  # (768,)

        # Unembedding matrix
        # In GPT-2, the unembedding is tied to the embedding (same weights transposed)
        # We want (model_dim, vocab_size) for projecting hidden states to logits
        # wte is (vocab_size, model_dim), so we transpose it
        unembed = wte.T  # (768, 50257)
        f.write("UNEMBEDDING\n")
        write_matrix(f, unembed)

    # Report file size
    file_size = output_file.stat().st_size
    print("\nExport complete!")
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"  Output: {output_path}")


def write_matrix(f, matrix: np.ndarray, precision: int = 8):
    """Write a 2D matrix to file in row-major order, space-separated."""
    for row in matrix:
        # Format each float with sufficient precision
        row_str = " ".join(f"{x:.{precision}g}" for x in row)
        f.write(row_str + "\n")


def write_vector(f, vector: np.ndarray, precision: int = 8):
    """Write a 1D vector as a single row."""
    row_str = " ".join(f"{x:.{precision}g}" for x in vector)
    f.write(row_str + "\n")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "models/gpt2.nfpt"
    export_gpt2_weights(output)
