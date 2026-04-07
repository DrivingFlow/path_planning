#!/usr/bin/env python3
"""Export TransformerModel to TorchScript for C++ LibTorch inference.

Usage:
    python3 export_torchscript.py \
        --weights /home/unitree/map_updater/src/grid_goat_model_map_new_v2.pth \
        --output  /home/unitree/path_planning/src/path_planning/models/model_scripted.pt

The exported model accepts:
    x_grids:  (1, 5, 2, 201, 201)  float16 on CUDA
    x_motion: (1, 5, 2)            float16 on CUDA
and returns:
    output:   (1, 5, 1, 201, 201)  float16 on CUDA
"""
import sys
import os
import argparse

import torch
torch.backends.cudnn.enabled = False

sys.path.insert(0, "/home/unitree/map_updater/src")
from TransformerModel import TransformerModel


def main():
    parser = argparse.ArgumentParser(description="Export TransformerModel to TorchScript")
    parser.add_argument("--weights", type=str,
                        default="/home/unitree/map_updater/src/grid_goat_model_map_new_v2.pth")
    parser.add_argument("--output", type=str,
                        default="/home/unitree/path_planning/src/path_planning/models/model_scripted.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading weights from {args.weights} ...")
    model = TransformerModel(grid_h=201, grid_w=201, motion_dim=2, num_decoder_layers=4)
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device(args.device)
    use_half = (device.type == "cuda")

    if use_half:
        model = model.half()
    model = model.to(device)

    dtype = torch.float16 if use_half else torch.float32
    dummy_grids = torch.randn(1, 5, 2, 201, 201, device=device, dtype=dtype)
    dummy_motion = torch.randn(1, 5, 2, device=device, dtype=dtype)

    # Run the original model for reference output
    with torch.no_grad():
        ref_out = model(dummy_grids, dummy_motion)
    print(f"Reference output shape: {ref_out.shape}, dtype: {ref_out.dtype}")

    # torch.jit.script fails on *s.shape[1:] unpacking in skip connections.
    # torch.jit.trace works for the fixed inference path (targets=None, no teacher forcing).
    # check_trace=False because fp16 non-determinism triggers the strict checker.
    print("Exporting with torch.jit.trace (inference path only) ...")
    scripted = torch.jit.trace(model, (dummy_grids, dummy_motion), check_trace=False)
    print("torch.jit.trace succeeded.")

    # Verify scripted model matches
    with torch.no_grad():
        scripted_out = scripted(dummy_grids, dummy_motion)

    max_diff = (ref_out - scripted_out).abs().max().item()
    print(f"Max abs difference between original and scripted: {max_diff:.6e}")
    if max_diff > 1e-2:
        print("WARNING: large difference, check model correctness!")
    else:
        print("Outputs match within tolerance.")

    # Save
    scripted.save(args.output)
    print(f"Saved TorchScript model to {args.output}")

    # Verify loading
    loaded = torch.jit.load(args.output, map_location=device)
    with torch.no_grad():
        loaded_out = loaded(dummy_grids, dummy_motion)
    reload_diff = (ref_out - loaded_out).abs().max().item()
    print(f"Reload verification max diff: {reload_diff:.6e}")
    print("Export complete.")


if __name__ == "__main__":
    main()
