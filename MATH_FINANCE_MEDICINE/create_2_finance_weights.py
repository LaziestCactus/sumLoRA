#!/usr/bin/env python3
import pickle
import numpy as np

# Paths to input LoRA pickles
MATH_MED_DEFAULT = "../MATH_MEDICINE/default_mathMed_lora_weights.pkl"
FINANCE_LORA     = "../FINANCE/finance_lora_weights.pkl"

# Load
with open(MATH_MED_DEFAULT, "rb") as f:
    mathmed = pickle.load(f)
with open(FINANCE_LORA, "rb") as f:
    fin = pickle.load(f)

# Combine: default_mathMed + fin
combined = {}
for name, arr in mathmed.items():
    combined[name] = np.array(arr, copy=True)
for name, arr in fin.items():
    if name in combined:
        combined[name] += np.array(arr)
    else:
        combined[name] = np.array(arr)

# Save
OUTPUT = "2_finance_lora_weights.pkl"
with open(OUTPUT, "wb") as f:
    pickle.dump(combined, f)

print(f"Saved combined Math+Meicine+Finance LoRA weights to {OUTPUT}")

