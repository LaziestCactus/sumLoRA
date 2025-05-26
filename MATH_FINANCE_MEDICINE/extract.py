#!/usr/bin/env python3
import pickle
import torch
from transformers import GPT2LMHeadModel
from peft import PeftModel

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "gpt2"
LORA_DIR     = "./combined_lora_model"
OUTPUT_PKL   = "temp.pkl"

# ─── DEVICE ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 1. LOAD BASE + PEFT (LoRA) MODEL ──────────────────────────────────────────
base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
peft_model = PeftModel.from_pretrained(base_model, LORA_DIR)
peft_model.to(device)
peft_model.eval()

# ─── 2. EXTRACT LO‐RA “DIFFERENCE” WEIGHTS ────────────────────────────────────
lora_weights = {
    name: param.detach().cpu().numpy()
    for name, param in peft_model.named_parameters()
    if "lora" in name
}

# ─── 3. SAVE TO PICKLE ─────────────────────────────────────────────────────────
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(lora_weights, f)

print(f"✔️  Saved {len(lora_weights)} LoRA parameter arrays to {OUTPUT_PKL}")

