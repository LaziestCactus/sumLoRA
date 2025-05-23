from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
import pickle
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from datasets import load_dataset
import matplotlib as plt
import os

# ─── Config ─────────────────────────────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
max_length = 128
batch_size = 8

# ─── Load & split finance QA dataset ────────────────────────────────────────────
financial_dataset = load_dataset("itzme091/financial-qa-10K-modified")
train_val_split    = financial_dataset["train"].train_test_split(test_size=0.2, seed=42)
val_test_split     = train_val_split["test"].train_test_split(test_size=0.5, seed=42)

train_data      = train_val_split["train"]
validation_data = val_test_split["train"]
test_data       = val_test_split["test"]

# ─── Prepare text lists ─────────────────────────────────────────────────────────
training_question   = list(train_data["question"])
training_answer     = list(train_data["answer"])
validation_question = list(validation_data["question"])
validation_answer   = list(validation_data["answer"])
test_question       = list(test_data["question"])
test_answer         = list(test_data["answer"])

print("Example train question:", training_question[0])
print("Example validation question:", validation_question[0])
print("Example test question:", test_question[0])

# ─── Dataset class ──────────────────────────────────────────────────────────────
class makeDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs  = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids":      self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels":         self.targets["input_ids"][idx],
        }

# ─── Build LoRA‐wrapped model ───────────────────────────────────────────────────
test_model = AutoModelForCausalLM.from_pretrained("gpt2")
lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 4,
    lora_alpha     = 64,
    lora_dropout   = 0.1,
    target_modules = ["c_attn", "c_proj"],
    modules_to_save= ["lm_head"]
)
test_model = get_peft_model(test_model, lora_config)

# ─── Reload LoRA weights ────────────────────────────────────────────────────────
with open("finance_lora_weights.pkl", "rb") as f:
    loaded_lora_weights = pickle.load(f)

count = 0
for name, param in test_model.named_parameters():
    if name in loaded_lora_weights:
        count += 1
        param.requires_grad = True
        print(f"Loading LoRA weight for: {name}")
        param.data.copy_(torch.tensor(loaded_lora_weights[name]))
print(f"Loaded {count} LoRA parameters")

# ─── Move model to device ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = test_model.to(device)

# ─── Tokenize all splits ────────────────────────────────────────────────────────
tokenizer.pad_token = tokenizer.eos_token

tokenized_training_question   = tokenizer(
    training_question, truncation=True, padding="max_length",
    max_length=max_length, return_tensors="pt"
)
tokenized_training_answer     = tokenizer(
    training_answer, truncation=True, padding=True,
    max_length=max_length, return_tensors="pt"
)
tokenized_validation_question = tokenizer(
    validation_question, truncation=True, padding="max_length",
    max_length=max_length, return_tensors="pt"
)
tokenized_validation_answer   = tokenizer(
    validation_answer, truncation=True, padding=True,
    max_length=max_length, return_tensors="pt"
)
tokenized_test_question       = tokenizer(
    test_question, truncation=True, padding="max_length",
    max_length=max_length, return_tensors="pt"
)
tokenized_test_answer         = tokenizer(
    test_answer, truncation=True, padding=True,
    max_length=max_length, return_tensors="pt"
)

print("Shapes:")
print("  Train Q:", tokenized_training_question["input_ids"].shape)
print("  Val   Q:", tokenized_validation_question["input_ids"].shape)
print("  Test  Q:", tokenized_test_question["input_ids"].shape)

# ─── Create Dataset objects ─────────────────────────────────────────────────────
train_dataset = makeDataset(tokenized_training_question, tokenized_training_answer)
val_dataset   = makeDataset(tokenized_validation_question, tokenized_validation_answer)
test_dataset  = makeDataset(tokenized_test_question, tokenized_test_answer)

print("Dataset sizes:")
print("  Train:", len(train_dataset))
print("  Val:  ", len(val_dataset))
print("  Test: ", len(test_dataset))

# ─── Evaluate LoRA‐augmented model ──────────────────────────────────────────────
model.eval()
total_loss  = 0.0
num_batches = 0

with torch.no_grad():
    for i in range(0, len(val_dataset), batch_size):
        if i + batch_size > len(val_dataset):
            break
        batch = val_dataset[i : i + batch_size]
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        total_loss  += outputs.loss.item()
        num_batches += 1

average_loss = total_loss / num_batches
perplexity   = torch.exp(torch.tensor(average_loss)).item()

print(f"Average Loss: {average_loss:.4f}")
print(f"Perplexity:   {perplexity:.4f}")

