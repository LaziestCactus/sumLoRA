import torch
from transformers import GPT2LMHeadModel
import pickle

# Load a default GPT-2 model from Hugging Face
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the multiplied LoRA weights from the pickle file
with open("my_map.pkl", "rb") as f:
    updated_dict = pickle.load(f)

# Apply the LoRA weights as a difference
for key, lora_weight in updated_dict.items():
    model_key = f"transformer.{key}"
    
    if model_key in model.state_dict():
        original_weight = model.state_dict()[model_key]
        
        # Transpose all attention and projection weights to match GPT-2's conventions
        # attn.c_proj might not need to transpose, I will test it out
        #if 'attn.c_attn.weight' in key or 'mlp.c_proj.weight' in key:
        lora_weight = lora_weight.T
        
        if original_weight.shape == lora_weight.shape:
            print(f"Updating {model_key} by adding LoRA weights")
            # Update the model's weights by adding the LoRA weights
            lora_alpha = 32
            lora_rank = 8
            lora_scale = lora_alpha/lora_rank
            model.state_dict()[model_key].copy_(original_weight + lora_weight*lora_scale)
        else:
            print(f"Shape mismatch for {model_key} even after transpose: original {original_weight.shape}, LoRA {lora_weight.shape}")
    else:
        print(f"Key {model_key} not found in GPT-2 model.")

# Save the modified model
model.save_pretrained("modified_gpt2_lora")