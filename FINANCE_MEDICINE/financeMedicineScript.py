from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pickle

# ─── DEVICE ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── HYPERPARAMETERS ───────────────────────────────────────────────────────
LoRA_RANK    = 4
LoRA_ALPHA   = 64
LoRA_DROPOUT = 0.15
MAX_LENGTH   = 128
BATCH_SIZE   = 16
NUM_EPOCHS   = 1

# ─── LOAD & PREPARE DATASETS ───────────────────────────────────────────────
finance_ds  = load_dataset("itzme091/financial-qa-10K-modified")["train"]
medicine_ds = load_dataset("keivalya/MedQuad-MedicalQnADataset")["train"]

finance = finance_ds.map(lambda x: {"question": x["question"], "answer": x["answer"]})
medicine = medicine_ds.map(lambda x: {"question": x["Question"], "answer": x["Answer"]})

questions = finance["question"] + medicine["question"]
answers   = finance["answer"]   + medicine["answer"]

# ─── TOKENIZER ────────────────────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

tokenized_inputs = tokenizer(
    questions,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)
tokenized_targets = tokenizer(
    answers,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

# ─── DATASET CLASS ────────────────────────────────────────────────────────
def build_dataset(inputs, targets):
    class QADataset(Dataset):
        def __init__(self, inputs, targets):
            self.inputs  = inputs
            self.targets = targets
        def __len__(self):
            return self.inputs["input_ids"].size(0)
        def __getitem__(self, idx):
            return {
                "input_ids":      self.inputs["input_ids"][idx],
                "attention_mask": self.inputs["attention_mask"][idx],
                "labels":         self.targets["input_ids"][idx],
            }
    return QADataset(inputs, targets)

full_ds = build_dataset(tokenized_inputs, tokenized_targets)

total      = len(full_ds)
train_size = int(0.5 * total)
val_size   = int(0.1 * total)
test_size  = total - train_size - val_size

train_ds, val_ds, test_ds = torch.utils.data.random_split(
    full_ds, [train_size, val_size, test_size]
)

# ─── CONFIGURE LoRA ───────────────────────────────────────────────────────
base_model = GPT2LMHeadModel.from_pretrained("gpt2")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LoRA_RANK,
    lora_alpha=LoRA_ALPHA,
    lora_dropout=LoRA_DROPOUT,
    target_modules=["c_attn", "c_proj"],
    modules_to_save=["lm_head"]
)
model = get_peft_model(base_model, lora_config).to(device)

# ─── TRAINING ARGUMENTS ───────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./finance_medicine_lora",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=50
)

# ─── SET UP & RUN TRAINER ─────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

print("Starting fine-tuning on Finance + Medicine data...")
trainer.train()

# ─── SAVE FULL LoRA-WRAPPED MODEL ────────────────────────────────────────
trainer.save_model("./finance_medicine_lora")

# ─── EXTRACT & SAVE ONLY LoRA WEIGHTS ──────────────────────────────────
peft_model   = PeftModel.from_pretrained(base_model, "./finance_medicine_lora").to(device)
lora_weights = {
    name: param.detach().cpu().numpy()
    for name, param in peft_model.named_parameters()
    if "lora_" in name
}

with open("financeMedicine_lora_weights.pkl", "wb") as f:
    pickle.dump(lora_weights, f)

print(f"Saved LoRA weights ({len(lora_weights)} layers) to financeMedicine_lora_weights.pkl")

