from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM
import pickle
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.amp import autocast
from datasets import load_dataset
import matplotlib as plt
import os

import datasets

# ─── Added missing tokenizer and max_length ───────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
max_length = 64 

ds = load_dataset("math_qa.py")
# Access the different splits
train_data      = ds['train']
validation_data = ds['validation']
test_data       = ds['test']

# Combine corresponding elements of "Problem" and "options"
training_question   = [p + " " + o for p, o in zip(train_data[:]["Problem"],      train_data[:]["options"])]
training_answer     = list(train_data[:]['Rationale'])
validation_question = [p + " " + o for p, o in zip(validation_data[:]["Problem"], validation_data[:]["options"])]
validation_answer   = list(validation_data[:]['Rationale'])
test_question       = [p + " " + o for p, o in zip(test_data[:]["Problem"],       test_data[:]["options"])]
test_answer         = list(test_data[:]['Rationale'])

print(training_question[0])
print(validation_answer[134])
print(test_answer[134])


class makeDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs  = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        input_ids      = self.inputs['input_ids'][idx]
        attention_mask = self.inputs['attention_mask'][idx]
        labels         = self.targets['input_ids'][idx]
        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels
        }


test_model = AutoModelForCausalLM.from_pretrained("gpt2")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)
test_model = get_peft_model(test_model, lora_config)

# To reload the LoRA weights for a model:
with open("math_lora_weights.pkl", "rb") as f:
    loaded_lora_weights = pickle.load(f)
count = 0
for name, param in test_model.named_parameters():
    if name in loaded_lora_weights:
        count += 1
        param.requires_grad = True
        print(f"Loading LoRA weight for: {name}")
        param.data.copy_(torch.tensor(loaded_lora_weights[name]))
print(count)


# ─── TESTING MODELS ────────────────────────────────────────────────────────────

# Fix NameError by using the LoRA-wrapped model directly
model  = test_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# tokenize dataset
tokenizer.pad_token = tokenizer.eos_token
tokenized_training_question   = tokenizer(training_question,   truncation=True, padding='max_length', return_tensors="pt", max_length=max_length)
tokenized_training_answer     = tokenizer(training_answer,     truncation=True, padding=True,         return_tensors="pt", max_length=max_length)
tokenized_validation_question = tokenizer(validation_question, truncation=True, padding='max_length', return_tensors="pt", max_length=max_length)
tokenized_validation_answer   = tokenizer(validation_answer,   truncation=True, padding=True,         return_tensors="pt", max_length=max_length)
tokenized_test_question       = tokenizer(test_question,       truncation=True, padding='max_length', return_tensors="pt", max_length=max_length)
tokenized_test_answer         = tokenizer(test_answer,         truncation=True, padding=True,         return_tensors="pt", max_length=max_length)

print(f"Tokenized Training Questions Shape: {tokenized_training_question['input_ids'].shape}")
print(f"Tokenized Training Answers Shape:   {tokenized_training_answer['input_ids'].shape}")

# Make sure it's divisible by batch size so last batch works fine
train_dataset = makeDataset(tokenized_training_question, tokenized_training_answer)
val_dataset   = makeDataset(tokenized_validation_question, tokenized_validation_answer)
test_dataset  = makeDataset(tokenized_test_question,       tokenized_test_answer)
print(f"Training dataset size:   {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size:       {len(test_dataset)}")

# ─── Evaluate the model ─────────────────────────────────────────────────────────
model.eval()
total_loss  = 0
num_batches = 0
batch_size  = 8
loss_hist   = []

with torch.no_grad():
    for i in range(0, len(val_dataset), batch_size):
        if i + batch_size > len(val_dataset):
            break

        batch = val_dataset[i : i + batch_size]
        # ── MOVE ALL TENSORS TO device ─────────────────────────
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss_hist.append(loss)
        total_loss  += loss.item()
        num_batches += 1

average_loss = total_loss / num_batches
perplexity   = torch.exp(torch.tensor(average_loss)).item()

print(f"Average Loss: {average_loss:.4f}")
print(f"Perplexity:   {perplexity:.4f}")

