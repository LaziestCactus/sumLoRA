import pickle

combined = {}  # sum loraA (medicine) and loraB (finance) before hand

# Load the two sets of LoRA weights from different files
with open("../MEDICINE/med_lora_weights.pkl", "rb") as f:
    med_lora_weights = pickle.load(f)
with open("../FINANCE/finance_lora_weights.pkl", "rb") as f:
    fin_lora_weights = pickle.load(f)

# Combine the med_lora_weights and fin_lora_weights into a single dictionary
for name, value in med_lora_weights.items():
    combined[name] = combined.get(name, 0) + value

for name, value in fin_lora_weights.items():
    combined[name] = combined.get(name, 0) + value

# (Optional) Inspect first couple of entries
print("First 2 combined layers:")
for i, (name, value) in enumerate(combined.items()):
    if i == 2:
        break
    print(f"{name}: {value.shape if hasattr(value, 'shape') else type(value)}")

print("\nFirst 2 medicine-only layers:")
for i, (name, value) in enumerate(med_lora_weights.items()):
    if i == 2:
        break
    print(f"{name}: {value.shape if hasattr(value, 'shape') else type(value)}")

# Save the combined LoRA weights
with open("financeAndMed_lora_weights.pkl", "wb") as f:
    pickle.dump(combined, f)

print(f"Saved combined Medicine+Finance LoRA weights ({len(combined)} layers) to financeAndMed_lora_weights.pkl")

