#!/usr/bin/env python3
import pickle
import numpy as np

# Paths to input LoRA pickles
FIN_MED_DEFAULT = "../FINANCE_MEDICINE/default_finMed_lora_weights.pkl"
MATH_LORA       = "../MATH/math_lora_weights.pkl"

# Load
with open(FIN_MED_DEFAULT, "rb") as f:
    finmed = pickle.load(f)
with open(MATH_LORA, "rb") as f:
    math = pickle.load(f)

# Combine: default_finMed + math
combined = {}
for name, arr in finmed.items():
    combined[name] = np.array(arr, copy=True)
for name, arr in math.items():
    if name in combined:
        combined[name] += np.array(arr)
    else:
        combined[name] = np.array(arr)

# Save
OUTPUT = "2_math_lora_weights.pkl"
with open(OUTPUT, "wb") as f:
    pickle.dump(combined, f)

print(f"Saved combined Finance+Medicine+Math LoRA weights to {OUTPUT}")

