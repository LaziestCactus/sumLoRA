import pickle
import torch

# ── CONFIG ───────────────────────────────────────────────────────────────────
LORA_PKL     = "temp.pkl"
OUTPUT_PKL   = "finance_lora_weights.pkl"
LORA_ALPHA   = 64  # must match training config
LORA_RANK    = 8   # must match training config

# ── 1. Load raw LoRA weights ──────────────────────────────────────────────────
with open(LORA_PKL, "rb") as f:
    lora_state: dict = pickle.load(f)

# ── 2. Build combined (scaled B @ A) weights ──────────────────────────────────
combined = {}
scaling = LORA_ALPHA / LORA_RANK

for key_A, weight_A in lora_state.items():
    if ".lora_A.default" not in key_A:
        continue

    key_B = key_A.replace(".lora_A.default", ".lora_B.default")
    weight_B = lora_state.get(key_B)
    if weight_B is None:
        continue

    A = torch.tensor(weight_A)
    B = torch.tensor(weight_B)

    # Apply scaling here
    delta = (B @ A) * scaling

    base_key = key_A.replace("base_model.model.", "")
    base_key = base_key.replace(".lora_A.default", "")

    combined[base_key] = delta

# ── 3. Save combined weights ──────────────────────────────────────────────────
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(combined, f)

print(f"✅ Saved {len(combined)} scaled LoRA deltas to '{OUTPUT_PKL}'")

