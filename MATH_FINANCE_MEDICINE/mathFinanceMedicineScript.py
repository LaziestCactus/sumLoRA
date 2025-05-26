#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# -------------------
# Hyperparameters
# -------------------
MODEL_NAME   = "gpt2"
LORA_RANK    = 4
LORA_ALPHA   = 64
LORA_DROPOUT = 0.1
MAX_LENGTH   = 64
BATCH_SIZE   = 16
NUM_EPOCHS   = 1
OUTPUT_DIR   = "./combined_lora_model"
SEED         = 42

# -------------------
# Device
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------
# Load base model + LoRA config
# -------------------
base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["c_attn", "c_proj"],
)
model = get_peft_model(base_model, lora_cfg).to(device)

# -------------------
# Load and split Finance data (80/10/10)
# -------------------
fin_full   = load_dataset("itzme091/financial-qa-10K-modified", split="train")
s1         = fin_full.train_test_split(test_size=0.2, seed=SEED)
fin_train  = s1["train"]
s2         = s1["test"].train_test_split(test_size=0.5, seed=SEED)
fin_val    = s2["train"]

# -------------------
# Load and split Medicine data (0-14999 train, 15000-15999 val)
# -------------------
med_full   = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
med_train  = med_full.select(range(0, 15000))
med_val    = med_full.select(range(15000, 16000))

# -------------------
# Load Math data (built-in splits)
# -------------------
math_ds    = load_dataset("math_qa.py")
math_train = math_ds["train"]
math_val   = math_ds.get("validation", math_train.train_test_split(test_size=0.1, seed=SEED)["test"])

# -------------------
# Tokenizer
# -------------------
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# -------------------
# Dataset class
# -------------------
class QADataset(Dataset):
    def __init__(self, questions, answers):
        enc_q = tokenizer(
            list(questions),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        enc_a = tokenizer(
            list(answers),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        self.input_ids      = enc_q.input_ids
        self.attention_mask = enc_q.attention_mask
        self.labels         = enc_a.input_ids

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }

# -------------------
# Build train & validation datasets
# -------------------
train_ds = ConcatDataset([
    QADataset(fin_train["question"], fin_train["answer"]),
    QADataset(med_train["Question"], med_train["Answer"]),
    QADataset(math_train["Problem"], math_train["Rationale"]),
])
val_ds = ConcatDataset([
    QADataset(fin_val["question"], fin_val["answer"]),
    QADataset(med_val["Question"], med_val["Answer"]),
    QADataset(math_val["Problem"], math_val["Rationale"]),
])

print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

# -------------------
# Training arguments
# -------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=50,
)

# -------------------
# Trainer & training
# -------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

print("▶️  Starting LoRA training on Finance, Medicine, and Math…")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"✔️  LoRA model saved to {OUTPUT_DIR}")

