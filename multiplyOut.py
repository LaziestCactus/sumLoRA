import pickle
import torch
import numpy as np

# Initialize a dictionary to store LoRA weights
lora_weights = {}

# To reload the LoRA weights for a model:
with open("med_lora_weights.pkl", "rb") as f:
    lora_weights = pickle.load(f)
    
    
# Updated dictionary after multiplication
updated_dict = {}

# Iterate through the dictionary to find pairs and perform multiplication
for key, weight_A in lora_weights.items():
    if 'lora_A.default' in key:
        # Create the corresponding lora_B key
        key_B = key.replace('lora_A.default', 'lora_B.default')
        
        # Get the matching lora_B weight
        weight_B = lora_weights.get(key_B)
        
        if weight_B is not None:
            # Perform matrix multiplication between weight_A and weight_B
            weight_A = torch.tensor(weight_A)
            weight_B = torch.tensor(weight_B)
            weight_combined = torch.matmul(weight_B, weight_A)
            
            # Remove 'lora_A.default' from the key for the final naming
            new_key = key.replace('.lora_A.default', '')
            new_key = new_key.replace('base_model.model.transformer.', '')
            
            # Add the new key and combined weight to the updated dictionary
            updated_dict[new_key] = weight_combined
            #print(new_key)

# # The updated_dict now contains the multiplied matrices with the standard GPT-2 keys
# # Print the key and the shape of the combined weights for the first few entries
# for i, (key, value) in enumerate(updated_dict.items()):
#     print(f"Key: {key}")
#     print(f"Shape: {value.shape}")
#     if i == 2:  # Stop after printing the first 3 entries
#         break

with open('my_map.pkl', 'wb') as f:
    pickle.dump(updated_dict, f)
