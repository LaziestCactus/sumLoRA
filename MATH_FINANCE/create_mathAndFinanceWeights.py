import pickle

combined = {}  # sum loraA (math) and loraB (finance) before hand

# Load the two sets of LoRA weights from different files
with open("../MATH/math_lora_weights.pkl", "rb") as f:
    math_lora_weights = pickle.load(f)
with open("../FINANCE/finance_lora_weights.pkl", "rb") as f:
    fin_lora_weights = pickle.load(f)

# Combine the math_lora_weights and fin_lora_weights into a single dictionary
for name, value in math_lora_weights.items():
    combined[name] = combined.get(name, 0) + value

for name, value in fin_lora_weights.items():
    combined[name] = combined.get(name, 0) + value

# (Optional) Inspect first couple of entries
print("First 2 combined layers:")
for i, (name, value) in enumerate(combined.items()):
    if i == 2:
        break
    shape = value.shape if hasattr(value, "shape") else type(value)
    print(f"{name}: {shape}")

print("\nFirst 2 math-only layers:")
for i, (name, value) in enumerate(math_lora_weights.items()):
    if i == 2:
        break
    shape = value.shape if hasattr(value, "shape") else type(value)
    print(f"{name}: {shape}")

# Save the combined LoRA weights
with open("mathAndFin_lora_weights.pkl", "wb") as f:
    pickle.dump(combined, f)

print(f"Saved combined Math+Finance LoRA weights ({len(combined)} layers) to mathAndFin_lora_weights.pkl")

