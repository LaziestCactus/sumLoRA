# view_lora_weights.py

import pickle
import numpy as np

# Path to the pickle file
pickle_file = "math_lora_weights.pkl"

# Load the pickle file
with open(pickle_file, "rb") as f:
    lora_weights = pickle.load(f)

# Print contents
print(f"Loaded {len(lora_weights)} LoRA layers from '{pickle_file}':\n")

for name, weights in lora_weights.items():
    print(f"Layer: {name}")
    print(f"Shape: {np.array(weights).shape}")
    print(f"Values (first 5 elements flattened): {np.array(weights).flatten()[:5]}\n")

