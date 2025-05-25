import pickle
combined = {} #sum loraA and loraB before hand

# Load the two sets of LoRA weights from different files
with open("../MEDICINE/med_lora_weights.pkl", "rb") as f:
    med_lora_weights = pickle.load(f)
with open("../MATH/math_lora_weights.pkl", "rb") as f:
    math_lora_weights = pickle.load(f)

# Combine the med_lora_weights and math_lora_weights into a single dictionary
for name, value in med_lora_weights.items():
    if name in combined:
        combined[name] = combined[name] + value  # Add weights together
    else:
        combined[name] = value  # Initialize the value if not present

for name, value in math_lora_weights.items():
    if name in combined:
        combined[name] = combined[name] + value  # Add weights together
    else:
        combined[name] = value  # Initialize the value if not present

# Print the combined dictionary and check contents
# Print the first 2 items from the combined dictionary
for i, (name, value) in enumerate(combined.items()):
    if i == 2:
        break
    print(f"{name}: {value}")

for i, (name, value) in enumerate(med_lora_weights.items()):
    if i == 2:
        break
    print(f"{name}: {value}")

with open("mathAndMed_lora_weights.pkl", "wb") as f:
    pickle.dump(combined, f)

# # Save the LoRA weights to a file
# with open("lora_weights.pkl", "wb") as f:
#     pickle.dump(lora_weights, f)
    
