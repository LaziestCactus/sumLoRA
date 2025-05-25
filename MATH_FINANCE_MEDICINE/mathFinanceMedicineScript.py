import os
import pickle

import torch
from torch.utils.data import Dataset
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
MODEL_NAME    = "gpt2"
LORA_RANK     = 4
LORA_ALPHA    = 64
LORA_DROPOUT  = 0.1
MAX_LENGTH    = 64
BATCH_SIZE    = 16
NUM_EPOCHS    = 1

MATH_WEIGHTS  = "4_math_lora_weights.pkl"   # math‐trained LoRA weights
OUTPUT_DIR    = "./combined_lora_model"
DIFF_WEIGHTS  = "lora_difference.pkl"      # where to save the difference

# -------------------
# Device
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------
# Load base model + LoRA
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
# Inject Math LoRA weights
# -------------------
with open(MATH_WEIGHTS, "rb") as f:
    math_lora = pickle.load(f)

for name, param in model.named_parameters():
    if name in math_lora:
        param.data = torch.tensor(math_lora[name], device=param.device)

print("Math LoRA weights loaded")

# -------------------
# Load & combine datasets
# -------------------
fin_ds  = load_dataset("itzme091/financial-qa-10K-modified")["train"]
med_ds  = load_dataset("keivalya/MedQuad-MedicalQnADataset")["train"]
math_ds = load_dataset("math_qa.py")["train"]

fin = fin_ds.map(lambda x: {"question": x["question"], "answer": x["answer"]})
med = med_ds.map(lambda x: {"question": x["Question"],  "answer": x["Answer"]})
math = math_ds.map(lambda x: {"question": x["Problem"],   "answer": x["Rationale"]})

questions = fin["question"] + med["question"] + math["question"]
answers   = fin["answer"]   + med["answer"]   + math["answer"]

# -------------------
# Tokenization
# -------------------
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

enc_q = tokenizer(
    questions,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)
enc_a = tokenizer(
    answers,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

# -------------------
# Dataset class
# -------------------
class QADataset(Dataset):
    def __init__(self, q, a):
        self.q = q
        self.a = a
    def __len__(self):
        return self.q["input_ids"].size(0)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.q["input_ids"][idx],
            "attention_mask": self.q["attention_mask"][idx],
            "labels":         self.a["input_ids"][idx],
        }

full_ds = QADataset(enc_q, enc_a)

# -------------------
# Train/Val/Test split
# -------------------
n        = len(full_ds)
n_train  = int(0.8 * n)
n_val    = int(0.1 * n)
n_test   = n - n_train - n_val

train_ds, val_ds, test_ds = torch.utils.data.random_split(
    full_ds,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

print(f"Sizes -> train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

# -------------------
# Training arguments
# -------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="no",
    save_strategy="no",
)

# -------------------
# Trainer
# -------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

# -------------------
# Fine-tune
# -------------------
trainer.train()
model.save_pretrained(OUTPUT_DIR)
print("Fine-tuning complete")

# -------------------
# Extract final LoRA weights
# -------------------
final_lora = {}
for name, param in model.named_parameters():
    if "lora" in name:
        final_lora[name] = param.detach().cpu().numpy()

print(f"Extracted {len(final_lora)} LoRA params")

# -------------------
# Compute & save difference
# -------------------
lora_diff = {}
for name, math_w in math_lora.items():
    if name in final_lora:
        lora_diff[name] = final_lora[name] - math_w

with open(DIFF_WEIGHTS, "wb") as f:
    pickle.dump(lora_diff, f)

print(f"LoRA difference saved to {DIFF_WEIGHTS}")

