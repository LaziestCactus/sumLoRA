#!/usr/bin/env python3
import pickle
import torch
import math
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ─── 1. Config ───────────────────────────────────────────────────────────────
DELTA_FILE   = "2_medicine_lora_weights.pkl"#"2_math_lora_weights.pkl"#"2_finance_lora_weights.pkl"#"default_all3_lora_weights.pkl"#"mathFinMed_lora_weights.pkl"
MODEL_NAME   = "gpt2"
BATCH_SIZE   = 8
MAX_LENGTH   = 64
SAMPLE_SIZE  = 50  # examples per domain
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 2. Load base GPT-2 and tokenizer ────────────────────────────────────────
model     = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ─── 3. Load and apply LoRA deltas ────────────────────────────────────────────
with open(DELTA_FILE, "rb") as f:
    delta_dict = pickle.load(f)

for name, param in model.named_parameters():
    if name not in delta_dict:
        continue
    delta = torch.tensor(delta_dict[name], dtype=param.dtype, device=param.device)
    if delta.shape != param.shape:
        if delta.T.shape == param.shape:
            delta = delta.T
            print(f"⚠️  Transposed delta for '{name}'")
        else:
            raise ValueError(f"Shape mismatch for '{name}': model {tuple(param.shape)}, delta {tuple(delta.shape)}")
    param.data.add_(delta)

print(f"✅ Applied LoRA deltas from {DELTA_FILE}\n")

# ─── 4. Prepare test splits ───────────────────────────────────────────────────
# 4a. Math: HF built-in test split
math_ds = load_dataset("math_qa.py", split="test")
math_test = math_ds.shuffle(seed=SEED).select(range(min(SAMPLE_SIZE, len(math_ds))))
math_qs, math_as = math_test["Problem"], math_test["Rationale"]

# 4b. Finance: true unseen 10%
fin_full = load_dataset("itzme091/financial-qa-10K-modified", split="train")
tail     = fin_full.train_test_split(test_size=0.2, seed=SEED)["test"]
fin_test = tail.train_test_split(test_size=0.5, seed=SEED)["test"]
fin_test = fin_test.shuffle(seed=SEED).select(range(min(SAMPLE_SIZE, len(fin_test))))
fin_qs, fin_as = fin_test["question"], fin_test["answer"]

# 4c. Medicine: indices >=16000
med_full    = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
med_holdout = med_full.select(range(16000, len(med_full)))
med_test    = med_holdout.shuffle(seed=SEED).select(range(min(SAMPLE_SIZE, len(med_holdout))))
med_qs, med_as = med_test["Question"], med_test["Answer"]

# ─── 5. Combine & tokenize ───────────────────────────────────────────────────
test_qs = list(math_qs) + list(fin_qs) + list(med_qs)
test_as = list(math_as) + list(fin_as) + list(med_as)

enc_q = tokenizer(
    test_qs,
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)
enc_a = tokenizer(
    test_as,
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

# ─── 6. Dataset wrapper ───────────────────────────────────────────────────────
class QADataset(Dataset):
    def __init__(self, enc_q, enc_a):
        self.enc_q = enc_q
        self.enc_a = enc_a
    def __len__(self):
        return self.enc_q["input_ids"].size(0)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc_q["input_ids"][idx],
            "attention_mask": self.enc_q["attention_mask"][idx],
            "labels":         self.enc_a["input_ids"][idx],
        }

test_ds = QADataset(enc_q, enc_a)

# ─── 7. Evaluation loop ───────────────────────────────────────────────────────
model.eval()
total_loss = 0.0
batches    = 0
with torch.no_grad():
    for i in range(0, len(test_ds), BATCH_SIZE):
        if i + BATCH_SIZE > len(test_ds):
            break
        batch = [test_ds[j] for j in range(i, i + BATCH_SIZE)]
        ids   = torch.stack([b["input_ids"]      for b in batch]).to(DEVICE)
        mask  = torch.stack([b["attention_mask"] for b in batch]).to(DEVICE)
        labs  = torch.stack([b["labels"]         for b in batch]).to(DEVICE)
        out   = model(input_ids=ids, attention_mask=mask, labels=labs)
        total_loss += out.loss.item()
        batches    += 1

avg_loss   = total_loss / batches
perplexity = math.exp(avg_loss)

print(f"\n📊 Test Evaluation (50 each: Math, Finance, Medicine)")
print(f"Average Loss    : {avg_loss:.4f}")
print(f"Perplexity      : {perplexity:.2f}")

