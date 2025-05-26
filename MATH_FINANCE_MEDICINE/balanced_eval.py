#!/usr/bin/env python3
import torch
import math
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from peft import PeftModel

# ─── CONFIG ────────────────────────────────────────────────────────────────
MODEL_DIR        = "./combined_lora_model"
MAX_LENGTH       = 64
EVAL_BATCH_SIZE  = 8
SAMPLE_SIZE      = 50
SEED             = 42
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD TOKENIZER & MODEL ─────────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Set pad token to EOS so padding works
tokenizer.pad_token = tokenizer.eos_token

base_model = GPT2LMHeadModel.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, MODEL_DIR).to(DEVICE)
model.eval()

# ─── DATASET WRAPPER ───────────────────────────────────────────────────────
class QADataset(Dataset):
    def __init__(self, questions, answers):
        enc_q = tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        enc_a = tokenizer(
            answers,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        self.input_ids      = enc_q.input_ids
        self.attention_mask = enc_q.attention_mask
        # use labels with padding token ignored by default
        self.labels         = enc_a.input_ids

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }

# ─── PREPARE TEST SETS ──────────────────────────────────────────────────────
math_ds = load_dataset("math_qa.py", split="test")
math_test = math_ds.shuffle(seed=SEED).select(range(min(SAMPLE_SIZE, len(math_ds))))
math_qs, math_as = math_test["Problem"], math_test["Rationale"]

fin_full = load_dataset("itzme091/financial-qa-10K-modified", split="train")
fin_tail = fin_full.train_test_split(test_size=0.2, seed=SEED)["test"]
fin_test = fin_tail.train_test_split(test_size=0.5, seed=SEED)["test"]
fin_test = fin_test.shuffle(seed=SEED).select(range(min(SAMPLE_SIZE, len(fin_test))))
fin_qs, fin_as = fin_test["question"], fin_test["answer"]

med_full = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
med_holdout = med_full.select(range(16000, len(med_full)))
med_test = med_holdout.shuffle(seed=SEED).select(range(min(SAMPLE_SIZE, len(med_holdout))))
med_qs, med_as = med_test["Question"], med_test["Answer"]

# ─── BUILD EVAL DATASET ─────────────────────────────────────────────────────
eval_ds = ConcatDataset([
    QADataset(math_qs, math_as),
    QADataset(fin_qs,  fin_as),
    QADataset(med_qs,  med_as),
])

# ─── EVALUATION ─────────────────────────────────────────────────────────────
eval_args = TrainingArguments(
    output_dir                 = "./eval_all3",
    per_device_eval_batch_size = EVAL_BATCH_SIZE,
    do_train                   = False,
    do_eval                    = True,
)

trainer = Trainer(
    model        = model,
    args         = eval_args,
    eval_dataset = eval_ds,
)
metrics = trainer.evaluate()
loss = metrics["eval_loss"]
ppl  = math.exp(loss)

print(f"Eval loss     : {loss:.4f}")
print(f"Perplexity    : {ppl:.2f}")

