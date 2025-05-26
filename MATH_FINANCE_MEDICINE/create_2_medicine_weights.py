#!/usr/bin/env python3
import pickle
import numpy as np

# Paths to input LoRA pickles
MATH_FIN_DEFAULT = "../MATH_FINANCE/default_mathFin_lora_weights.pkl"
MED_LORA         = "../MEDICINE/med_lora_weights.pkl"

# Load
with open(MATH_FIN_DEFAULT, "rb") as f:
    mathfin = pickle.load(f)
with open(MED_LORA, "rb") as f:
    med = pickle.load(f)

# Combine: default_mathFin + med
combined = {}
for name, arr in mathfin.items():
    combined[name] = np.array(arr, copy=True)
for name, arr in med.items():
    if name in combined:
        combined[name] += np.array(arr)
    else:
        combined[name] = np.array(arr)

# Save
OUTPUT = "2_medicine_lora_weights.pkl"
with open(OUTPUT, "wb") as f:
    pickle.dump(combined, f)

print(f"Saved combined Math+Finance+Medicine LoRA weights to {OUTPUT}")

