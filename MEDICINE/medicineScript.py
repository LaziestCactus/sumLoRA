from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pickle
import os

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────────
LoRA_Rank    = 16
LoRA_Alpha   = 64
LoRA_Dropout = 0.3
max_length   = 64
batch_size   = 8
num_epochs   = 2
model_name   = "gpt2"
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD & PREPARE MEDQUAD DATASET ─────────────────────────────────────────────
ds = load_dataset("keivalya/MedQuad-MedicalQnADataset")
training_data   = ds["train"][:15000]
validation_data = ds["train"][15000:16000]

training_question   = list(training_data["Question"])
training_answer     = list(training_data["Answer"])
validation_question = list(validation_data["Question"])
validation_answer   = list(validation_data["Answer"])

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

# ─── MODEL & LoRA SETUP ──────────────────────────────────────────────────────────
GPTmodel = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = LoRA_Rank,
    lora_alpha     = LoRA_Alpha,
    lora_dropout   = LoRA_Dropout,
    target_modules = ["c_attn", "c_proj"]
)

model = get_peft_model(GPTmodel, lora_config).to(device)

# ─── TOKENIZE ───────────────────────────────────────────────────────────────────
tok_train_q = tokenizer(
    training_question,
    truncation=True,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt"
)
tok_train_a = tokenizer(
    training_answer,
    truncation=True,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt"
)
tok_val_q = tokenizer(
    validation_question,
    truncation=True,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt"
)
tok_val_a = tokenizer(
    validation_answer,
    truncation=True,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt"
)

train_dataset = makeDataset(tok_train_q, tok_train_a)
val_dataset   = makeDataset(tok_val_q,   tok_val_a)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# ─── TRAIN ───────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                = "./results",
    num_train_epochs          = num_epochs,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size  = batch_size,
    warmup_steps              = 2,
    weight_decay              = 0.01,
    logging_dir               = "./logs",
    logging_steps             = 50,
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
)

print(f"Starting training on {device}")
trainer.train()
trainer.save_model("./medLora-model")
print("Fine-tuning complete, model saved to ./medLora-model")

# ─── EVALUATE FINETUNED LoRA MODEL ───────────────────────────────────────────────
model.eval()
total_loss, n_batches = 0.0, 0
with torch.no_grad():
    for i in range(0, len(val_dataset), batch_size):
        if i + batch_size > len(val_dataset): break
        batch = val_dataset[i : i+batch_size]
        inputs  = batch["input_ids"].to(device)
        masks   = batch["attention_mask"].to(device)
        labels  = batch["labels"].to(device)
        out     = model(input_ids=inputs, attention_mask=masks, labels=labels)
        total_loss += out.loss.item()
        n_batches  += 1

avg_loss   = total_loss / n_batches
perplexity = torch.exp(torch.tensor(avg_loss)).item()
print(f"After fine-tuning → Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

# ─── EXTRACT & SAVE LoRA WEIGHTS ────────────────────────────────────────────────
print("Extracting LoRA weights…")
base_model = AutoModelForCausalLM.from_pretrained(model_name)
lora_model = PeftModel.from_pretrained(base_model, "./medLora-model")

lora_weights = {}
count = 0
for name, param in lora_model.named_parameters():
    if "lora" in name:
        lora_weights[name] = param.detach().cpu().numpy()
        print(f" • {name} {tuple(param.shape)}")
        count += 1
print(f"Extracted {count} LoRA parameters")

with open("med_lora_weights.pkl", "wb") as f:
    pickle.dump(lora_weights, f)
print("LoRA weights saved to med_lora_weights.pkl")

# ─── RELOAD LoRA WEIGHTS & RE-EVALUATE ─────────────────────────────────────────
print("Reloading LoRA weights and re-evaluating…")
test_model = AutoModelForCausalLM.from_pretrained(model_name)
test_model = get_peft_model(test_model, lora_config).to(device)

with open("med_lora_weights.pkl", "rb") as f:
    loaded = pickle.load(f)
for name, param in test_model.named_parameters():
    if name in loaded:
        param.data.copy_(torch.tensor(loaded[name], device=device))
        param.requires_grad = True

test_model.eval()
total_loss, n_batches = 0.0, 0
with torch.no_grad():
    for i in range(0, len(val_dataset), batch_size):
        if i + batch_size > len(val_dataset): break
        batch = val_dataset[i : i+batch_size]
        inputs  = batch["input_ids"].to(device)
        masks   = batch["attention_mask"].to(device)
        labels  = batch["labels"].to(device)
        out     = test_model(input_ids=inputs, attention_mask=masks, labels=labels)
        total_loss += out.loss.item()
        n_batches  += 1

avg_loss2   = total_loss / n_batches
perplexity2 = torch.exp(torch.tensor(avg_loss2)).item()
print(f"After reload → Loss: {avg_loss2:.4f}, Perplexity: {perplexity2:.4f}")

