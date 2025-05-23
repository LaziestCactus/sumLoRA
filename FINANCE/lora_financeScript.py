from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pickle
import os

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────────
LoRA_Rank    = 4
LoRA_Alpha   = 64
LoRA_Dropout = 0.1
max_length   = 128
batch_size   = 8
num_epochs   = 1
model_name   = "gpt2"
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD & SPLIT FINANCE QA DATASET ────────────────────────────────────────────
financial = load_dataset("itzme091/financial-qa-10K-modified")["train"]
tv_split = financial.train_test_split(test_size=0.2, seed=42)
vt_split = tv_split["test"].train_test_split(test_size=0.5, seed=42)

train_data      = tv_split["train"]
validation_data = vt_split["train"]
test_data       = vt_split["test"]

# ─── PREPARE TEXT LISTS ─────────────────────────────────────────────────────────
train_q = list(train_data["question"])
train_a = list(train_data["answer"])
val_q   = list(validation_data["question"])
val_a   = list(validation_data["answer"])
test_q  = list(test_data["question"])
test_a  = list(test_data["answer"])

# ─── TOKENIZER & LoRA‐WRAPPED MODEL SETUP ────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = GPT2LMHeadModel.from_pretrained(model_name)
lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = LoRA_Rank,
    lora_alpha     = LoRA_Alpha,
    lora_dropout   = LoRA_Dropout,
    target_modules = ["c_attn", "c_proj"]
)
model = get_peft_model(base_model, lora_config).to(device)

# ─── TOKENIZATION HELPER ─────────────────────────────────────────────────────────
def tokenize(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

tok_train_q = tokenize(train_q)
tok_train_a = tokenize(train_a)
tok_val_q   = tokenize(val_q)
tok_val_a   = tokenize(val_a)
tok_test_q  = tokenize(test_q)
tok_test_a  = tokenize(test_a)

# ─── DATASET CLASS ───────────────────────────────────────────────────────────────
class FinancialQADataset(Dataset):
    def __init__(self, enc_q, enc_a):
        self.enc_q = enc_q
        self.enc_a = enc_a
    def __len__(self):
        return len(self.enc_q["input_ids"])
    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc_q["input_ids"][idx],
            "attention_mask": self.enc_q["attention_mask"][idx],
            "labels":         self.enc_a["input_ids"][idx],
        }

train_ds = FinancialQADataset(tok_train_q, tok_train_a)
val_ds   = FinancialQADataset(tok_val_q,   tok_val_a)
test_ds  = FinancialQADataset(tok_test_q,  tok_test_a)

# ─── FINE-TUNE WITH LoRA ─────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                = "./financeresults",
    num_train_epochs          = num_epochs,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size  = batch_size,
    warmup_steps              = 2,
    weight_decay              = 0.01,
    logging_dir               = "./financelogs",
    logging_steps             = 50,
    evaluation_strategy       = "no"
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_ds
)

print("Starting LoRA fine-tuning…")
trainer.train()
trainer.save_model("./financeLora-model")
print("✔️  Model saved to ./financeLora-model")

# ─── SAVE LoRA WEIGHTS ───────────────────────────────────────────────────────────
print("Extracting LoRA weights…")
lora_weights = {}
count = 0
for name, param in model.named_parameters():
    if name.startswith("base_model") and "lora" in name:
        lora_weights[name] = param.detach().cpu().numpy()
        count += 1
print(f"  Collected {count} LoRA tensors")
with open("finance_lora_weights.pkl", "wb") as f:
    pickle.dump(lora_weights, f)

# ─── EVALUATE FINETUNED MODEL ───────────────────────────────────────────────────
eval_loader = DataLoader(val_ds, batch_size=batch_size)
model.eval()
total_loss, n_batches = 0.0, 0
with torch.no_grad():
    for batch in eval_loader:
        inputs = batch["input_ids"].to(device)
        masks  = batch["attention_mask"].to(device)
        labs   = batch["labels"].to(device)
        out    = model(input_ids=inputs, attention_mask=masks, labels=labs)
        total_loss += out.loss.item()
        n_batches  += 1

avg_loss = total_loss / n_batches
perplexity = torch.exp(torch.tensor(avg_loss)).item()
print(f"Fine-tuned Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

# ─── RELOAD & RE-EVALUATE LoRA WEIGHTS ───────────────────────────────────────────
print("Reloading LoRA weights…")
reload_base = AutoModelForCausalLM.from_pretrained(model_name)
reload_model = get_peft_model(reload_base, lora_config).to(device)

with open("finance_lora_weights.pkl", "rb") as f:
    loaded = pickle.load(f)
for name, param in reload_model.named_parameters():
    if name in loaded:
        param.data.copy_(torch.tensor(loaded[name], device=device))

reload_model.eval()
total_loss, n_batches = 0.0, 0
with torch.no_grad():
    for batch in eval_loader:
        inputs = batch["input_ids"].to(device)
        masks  = batch["attention_mask"].to(device)
        labs   = batch["labels"].to(device)
        out    = reload_model(input_ids=inputs, attention_mask=masks, labels=labs)
        total_loss += out.loss.item()
        n_batches  += 1

avg_loss2 = total_loss / n_batches
perplexity2 = torch.exp(torch.tensor(avg_loss2)).item()
print(f"Reloaded Loss: {avg_loss2:.4f}, Perplexity: {perplexity2:.4f}")

