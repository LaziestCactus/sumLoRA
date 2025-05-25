from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pickle
import os

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────────
LoRA_Rank    = 4 
LoRA_Alpha   = 64
LoRA_Dropout = 0.25
max_length   = 128   # longest tokenized sequence
batch_size   = 4
num_epochs   = 2
model_name   = "gpt2"
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD & SPLIT FINANCE DATASET ───────────────────────────────────────────────
financial_dataset = load_dataset("itzme091/financial-qa-10K-modified")
# 80/10/10 split
train_val_split = financial_dataset["train"].train_test_split(test_size=0.2, seed=42)
val_test_split  = train_val_split["test"].train_test_split(test_size=0.5, seed=42)

train_data      = train_val_split["train"]
validation_data = val_test_split["train"]
test_data       = val_test_split["test"]

# ─── TOKENIZER & MODEL SETUP ─────────────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = GPT2LMHeadModel.from_pretrained(model_name)
lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = LoRA_Rank,
    lora_alpha     = LoRA_Alpha,
    lora_dropout   = LoRA_Dropout,
    target_modules = ["c_attn", "c_proj"],
    modules_to_save= ["lm_head"]
)
model = get_peft_model(base_model, lora_config).to(device)

# ─── DATASET CLASS ───────────────────────────────────────────────────────────────
class FinancialQADataset(Dataset):
    def __init__(self, enc_inputs, enc_targets):
        self.enc_inputs  = enc_inputs
        self.enc_targets = enc_targets
    def __len__(self):
        return len(self.enc_inputs["input_ids"])
    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc_inputs["input_ids"][idx],
            "attention_mask": self.enc_inputs["attention_mask"][idx],
            "labels":         self.enc_targets["input_ids"][idx],
        }

# ─── TOKENIZATION HELPER ─────────────────────────────────────────────────────────
def tokenize_data(questions, answers):
    q = tokenizer(
        questions,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    a = tokenizer(
        answers,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    return q, a

tok_train_q, tok_train_a = tokenize_data(train_data["question"],      train_data["answer"])
tok_val_q,   tok_val_a   = tokenize_data(validation_data["question"], validation_data["answer"])
tok_test_q,  tok_test_a  = tokenize_data(test_data["question"],       test_data["answer"])

train_dataset = FinancialQADataset(tok_train_q, tok_train_a)
val_dataset   = FinancialQADataset(tok_val_q,   tok_val_a)
test_dataset  = FinancialQADataset(tok_test_q,  tok_test_a)

# ─── TRAINING ───────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir             = "./financeresults",
    num_train_epochs       = num_epochs,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size  = batch_size,
    warmup_steps           = 2,
    weight_decay           = 0.01,
    logging_dir            = "./financelogs",
    logging_steps          = 50,
    evaluation_strategy    = "epoch",
    save_strategy          = "epoch",
    save_total_limit       = 2,
    fp16                   = True,    # mixed precision
    remove_unused_columns  = False,
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
)

print(f"Starting LoRA fine-tuning on {device}…")
trainer.train()
trainer.save_model("./financeLora-model")
print("✔️  Fine-tuning complete and model saved to ./financeLora-model")

# ─── EVALUATION ────────────────────────────────────────────────────────────────
print("Evaluating on validation split…")
model.eval()
total_loss, n_batches = 0.0, 0
with torch.no_grad():
    for i in range(0, len(val_dataset), batch_size):
        if i + batch_size > len(val_dataset): break
        batch = val_dataset[i : i+batch_size]
        inputs = batch["input_ids"].to(device)
        masks  = batch["attention_mask"].to(device)
        labs   = batch["labels"].to(device)
        out    = model(input_ids=inputs, attention_mask=masks, labels=labs)
        total_loss += out.loss.item()
        n_batches  += 1

avg_loss   = total_loss / n_batches
perplexity = torch.exp(torch.tensor(avg_loss)).item()
print(f"Average Loss: {avg_loss:.4f}")
print(f"Perplexity:   {perplexity:.4f}")

# ─── EXTRACT & SAVE LoRA WEIGHTS ────────────────────────────────────────────────
print("Extracting LoRA weights…")
lora_model = PeftModel.from_pretrained(base_model, "./financeLora-model")
lora_weights = {}
count = 0
for name, param in lora_model.named_parameters():
    if "lora" in name:
        print(f" • {name} {tuple(param.shape)}")
        lora_weights[name] = param.detach().cpu().numpy()
        count += 1

pickle.dump(lora_weights, open("temp.pkl", "wb"))
print(f"Saved {count} LoRA parameter arrays to temp.pkl")

