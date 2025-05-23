#THIS DOES NOT WORK BUT IT'S NOT NEEDED
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.amp import autocast
from datasets import load_dataset
import matplotlib as plt
import matplotlib.pyplot as plt
import os
import sys
import pickle
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer






test_model = AutoModelForCausalLM.from_pretrained("gpt2")  # Make sure this is the correct model (gpt2, not gp3)
# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank of the low-rank adaptation matrices
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules = ["c_attn", "c_proj"]
)
# Prepare model for LoRA tuning
test_model = get_peft_model(test_model, lora_config)

# To reload the LoRA weights for a model:
with open("mathAndMed_lora_weights.pkl", "rb") as f:
    loaded_lora_weights = pickle.load(f)
count = 0
# Apply the LoRA weights to the model
for name, param in test_model.named_parameters():
    if name in loaded_lora_weights:
        count = count + 1
        param.requires_grad = True
        print(f"Loading LoRA weight for: {name}")
        param.data.copy_(torch.tensor(loaded_lora_weights[name]))
print(count)






#TESTING MODEL
model = test_model
# Evaluate the model
model.eval()
total_loss = 0
num_batches = 0
batch_size = 8  # Adjust based on your memory constraints
loss_hist = [] # Storing it, no use for now
with torch.no_grad():
    for i in range(0, len(val_dataset), batch_size):
        if(i+batch_size >= len(val_dataset)):
            break
        batch = val_dataset[i:i + batch_size]
        # Get input_ids and attention_mask from the batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask'] if 'attention_mask' in batch else None

        # Pass input_ids as labels for loss calculation
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels= batch['labels'])
        
        loss = outputs.loss
        loss_hist.append(loss)
        total_loss += loss.item()
        num_batches += 1

# Calculate average loss and perplexity
average_loss = total_loss / num_batches
perplexity = torch.exp(torch.tensor(average_loss)).item()

print(f"Average Loss: {average_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")
