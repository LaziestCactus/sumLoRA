import optuna
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset
from datasets import load_dataset

# ─── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "gpt2"
MAX_LENGTH   = 64
TRAIN_SLICE  = 15000
VAL_START    = 15000
VAL_END      = 16000
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load & tokenize MedQuad dataset ────────────────────────────────────────────
ds = load_dataset("keivalya/MedQuad-MedicalQnADataset")
train_data      = ds["train"][:TRAIN_SLICE]
validation_data = ds["train"][VAL_START:VAL_END]

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

training_question   = train_data["Question"]
training_answer     = train_data["Answer"]
validation_question = validation_data["Question"]
validation_answer   = validation_data["Answer"]

def tokenize(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

tok_train_q = tokenize(training_question)
tok_train_a = tokenize(training_answer)
tok_val_q   = tokenize(validation_question)
tok_val_a   = tokenize(validation_answer)

class makeDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs  = inputs
        self.targets = targets
    def __len__(self):
        return len(self.inputs["input_ids"])
    def __getitem__(self, idx):
        return {
            "input_ids":      self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels":         self.targets["input_ids"][idx],
        }

train_dataset = makeDataset(tok_train_q, tok_train_a)
val_dataset   = makeDataset(tok_val_q,   tok_val_a)

# ─── Objective function for Optuna ─────────────────────────────────────────────
def objective(trial):
    # clear CUDA cache to avoid fragmentation
    torch.cuda.empty_cache()

    # sample hyperparameters
    LoRA_Rank    = trial.suggest_categorical("LoRA_Rank",    [4, 8, 16])
    LoRA_Alpha   = trial.suggest_categorical("LoRA_Alpha",   [16, 32, 64])
    LoRA_Dropout = trial.suggest_float(     "LoRA_Dropout", 0.1, 0.3, step=0.05)
    batch_size   = trial.suggest_categorical("batch_size",   [8, 16])
    num_epochs   = trial.suggest_int(        "num_epochs",   1, 3)

    # build a fresh LoRA‐wrapped GPT2
    base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    lora_cfg   = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = LoRA_Rank,
        lora_alpha     = LoRA_Alpha,
        lora_dropout   = LoRA_Dropout,
        target_modules = ["c_attn", "c_proj"]
    )
    model = get_peft_model(base_model, lora_cfg).to(DEVICE)
    # (no gradient_checkpointing)

    # training arguments
    args = TrainingArguments(
        output_dir                   = f"./optuna_trial_{trial.number}",
        num_train_epochs             = num_epochs,
        per_device_train_batch_size  = batch_size,
        per_device_eval_batch_size   = batch_size,
        logging_steps                = 50,
        logging_dir                  = "./optuna_logs",
        fp16                         = True,
        remove_unused_columns        = False,  # keep label tensors
        save_strategy                = "no",
        disable_tqdm                 = True,
    )

    trainer = Trainer(
        model         = model,
        args          = args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset
    )

    # fine‐tune & evaluate
    trainer.train()
    metrics = trainer.evaluate()
    return metrics["eval_loss"]

# ─── Run the study ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=6)

    print("🔍 Best trial:")
    print(f"  Loss:   {study.best_value:.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

