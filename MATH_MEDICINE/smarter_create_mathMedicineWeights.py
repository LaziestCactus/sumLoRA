#!/usr/bin/env python3
"""
Take your MEDICINE & MATH LoRA pickles, orthogonalise via Gram–Schmidt,
and save a merged delta in 4_mathAndMed_GSmerged.pkl
"""

import torch
import pickle
from pathlib import Path

EPS = 1e-9  # numerical stability for dot/division


def gs_merge(delta_a: torch.Tensor, delta_b: torch.Tensor) -> torch.Tensor:
    """
    Orthogonalise delta_b w.r.t. delta_a, then add.
    Also rescales the orthogonal part so that the merged tensor
    has the mean Frobenius norm of delta_a & the orthogonal component.
    """
    # flatten for dot‐product
    a_flat = delta_a.flatten()
    b_flat = delta_b.flatten()

    # projection of b onto a
    proj_coeff = torch.dot(b_flat, a_flat) / (torch.dot(a_flat, a_flat) + EPS)
    b_perp = delta_b - proj_coeff * delta_a

    # energy balancing: target norm is average of norms
    norm_a = delta_a.norm()
    norm_bperp = b_perp.norm()
    target = 0.5 * (norm_a + norm_bperp)

    if norm_bperp > EPS:
        b_perp = b_perp * (target / norm_bperp)

    return delta_a + b_perp


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: dict, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    # your data paths
    med_path  = Path("../MEDICINE/4_med_lora_weights.pkl")
    math_path = Path("../MATH/4_math_lora_weights.pkl")
    out_path  = Path("4_mathAndMed_GSmerged.pkl")

    print(f"Loading MEDICINE LoRA deltas from {med_path}")
    delta_med  = load_pickle(med_path)
    print(f"Loading   MATH LoRA deltas from {math_path}")
    delta_math = load_pickle(math_path)

    merged = {}
    all_keys = set(delta_med.keys()) | set(delta_math.keys())
    for k in all_keys:
        w_med  = delta_med .get(k)
        w_math = delta_math.get(k)

        if w_med is None:
            merged[k] = w_math
        elif w_math is None:
            merged[k] = w_med
        else:
            merged[k] = gs_merge(w_med, w_math)

    # ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(merged, out_path)
    print(f"✔ Saved GS-merged LoRA deltas → {out_path}")

    # Quick sanity check: print norms of first few layers
    print("\nLayer norms (||med||, ||math||, ||merged||) for first 3 keys:")
    for i, k in enumerate(list(merged.keys())[:3]):
        nm = delta_med.get(k).norm().item()   if k in delta_med  else 0.0
        nx = delta_math.get(k).norm().item()  if k in delta_math else 0.0
        ny = merged[k].norm().item()
        print(f"  {k:30s}   ({nm:.4f}, {nx:.4f}) → {ny:.4f}")


if __name__ == "__main__":
    main()

