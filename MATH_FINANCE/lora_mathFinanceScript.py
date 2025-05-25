import pickle
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── CONFIG ───────────────────────────────────────────────────────────────────
DELTA_FILE = "mathAndFin_lora_weights.pkl"
MODEL_NAME = "gpt2"
BATCH_SIZE = 8
MAX_LENGTH = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Load base GPT-2 and tokenizer ─────────────────────────────────────────
model     = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ── 2. Load and apply Finance+Math LoRA deltas ────────────────────────────────
with open(DELTA_FILE, "rb") as f:
    delta_dict = pickle.load(f)

applied = 0
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
    applied += 1
print(f"\n✅ Applied deltas to {applied} parameters\n")

# ── 3. Load and preprocess Math+Finance dataset ───────────────────────────────
finance_ds = load_dataset("itzme091/financial-qa-10K-modified")["train"]
math_ds    = load_dataset("math_qa.py")["train"]

finance = finance_ds.map(lambda x: {"question": x["question"], "answer": x["answer"]})
math    = math_ds.map(lambda x: {"question": x["Problem"],  "answer": x["Rationale"]})

all_qs = finance["question"] + math["question"]
all_as = finance["answer"]   + math["answer"]

total     = len(all_qs)
train_end = int(0.8 * total)
val_end   = train_end + int(0.1 * total)

test_qs = all_qs[val_end:]
test_as = all_as[val_end:]

# ── 4. Tokenize test split ────────────────────────────────────────────────────
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

class QADataset(Dataset):
    def __init__(self, enc_q, enc_a):
        self.q = enc_q
        self.a = enc_a
    def __len__(self):
        return self.q["input_ids"].size(0)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.q["input_ids"][idx],
            "attention_mask": self.q["attention_mask"][idx],
            "labels":         self.a["input_ids"][idx],
        }

test_ds = QADataset(enc_q, enc_a)

# ── 5. Evaluate the merged model ──────────────────────────────────────────────
model.eval()
total_loss, batches = 0.0, 0
with torch.no_grad():
    for i in range(0, len(test_ds), BATCH_SIZE):
        if i + BATCH_SIZE > len(test_ds):
            break
        batch = [test_ds[j] for j in range(i, i + BATCH_SIZE)]
        ids   = torch.stack([b["input_ids"]      for b in batch]).to(DEVICE)
        mask  = torch.stack([b["attention_mask"] for b in batch]).to(DEVICE)
        labs  = torch.stack([b["labels"]         for b in batch]).to(DEVICE)

        outputs = model(input_ids=ids, attention_mask=mask, labels=labs)
        total_loss += outputs.loss.item()
        batches    += 1

avg_loss   = total_loss / batches
perplexity = torch.exp(torch.tensor(avg_loss)).item()

print("\n📊 Math + Finance Test Evaluation")
print(f"Average Loss : {avg_loss:.4f}")
print(f"Perplexity   : {perplexity:.4f}")

