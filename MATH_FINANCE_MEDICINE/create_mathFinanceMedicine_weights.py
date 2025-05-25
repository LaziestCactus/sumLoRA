import pickle

combined = {}  # sum LoRA weights from math, finance, and medicine

# Load the three sets of LoRA weights
with open("../MATH/math_lora_weights.pkl", "rb") as f:
    math_lora = pickle.load(f)
with open("../FINANCE/finance_lora_weights.pkl", "rb") as f:
    finance_lora = pickle.load(f)
with open("../MEDICINE/med_lora_weights.pkl", "rb") as f:
    med_lora = pickle.load(f)

# Combine math + finance
for name, value in math_lora.items():
    combined[name] = combined.get(name, 0) + value
for name, value in finance_lora.items():
    combined[name] = combined.get(name, 0) + value

# Add medicine on top
for name, value in med_lora.items():
    combined[name] = combined.get(name, 0) + value

# (Optional) Inspect first couple of entries
print("First 2 combined layers:")
for i, (name, value) in enumerate(combined.items()):
    if i == 2:
        break
    shape = value.shape if hasattr(value, "shape") else type(value)
    print(f"  {name}: {shape}")

# Save the combined LoRA weights
output_path = "mathFinMed_lora_weights.pkl"
with open(output_path, "wb") as f:
    pickle.dump(combined, f)

print(f"Saved combined Math+Finance+Medicine LoRA weights ({len(combined)} layers) to {output_path}")

