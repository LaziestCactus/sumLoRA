from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Dataset
import optuna
from datasets import load_dataset

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
financial_dataset = load_dataset("itzme091/financial-qa-10K-modified")

# Split dataset into train (80%), validation (10%), test (10%)
train_val_split = financial_dataset["train"].train_test_split(test_size=0.2, seed=42)
val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)

train_data = train_val_split["train"]
validation_data = val_test_split["train"]
test_data = val_test_split["test"]

# Custom Dataset Class
class FinancialQADataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.targets['input_ids'][idx]
        }

# Load model and tokenizer
model_name = "gpt2"
GPTmodel = GPT2LMHeadModel.from_pretrained(model_name).to(device)
GPTmodel.gradient_checkpointing_enable()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Tokenization
tokenizer.pad_token = tokenizer.eos_token
max_length = 128

def tokenize_data(questions, answers):
    tokenized_questions = tokenizer(questions, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
    tokenized_answers = tokenizer(answers, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
    return tokenized_questions, tokenized_answers

tokenized_train_question, tokenized_train_answer = tokenize_data(train_data["question"], train_data["answer"])
tokenized_val_question, tokenized_val_answer = tokenize_data(validation_data["question"], validation_data["answer"])

# Create dataset instances
train_dataset = FinancialQADataset(tokenized_train_question, tokenized_train_answer)
val_dataset = FinancialQADataset(tokenized_val_question, tokenized_val_answer)

def objective(trial):
    torch.cuda.empty_cache()  # Clear memory before training

    # Hyperparameters
    LoRA_Rank = trial.suggest_categorical('LoRA_Rank', [4, 8, 16])
    LoRA_Alpha = trial.suggest_categorical('LoRA_Alpha', [16, 32, 64])
    LoRA_Dropout = trial.suggest_float('LoRA_Dropout', 0.1, 0.3, step=0.05)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    num_epochs = trial.suggest_int('num_epochs', 1, 3)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LoRA_Rank,
        lora_alpha=LoRA_Alpha,
        lora_dropout=LoRA_Dropout,
        target_modules=["c_attn", "c_proj"],
        modules_to_save=["lm_head"]
    )

    model = get_peft_model(GPT2LMHeadModel.from_pretrained(model_name), lora_config).to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_trial_{trial.number}',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=2,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",  # Save every epoch
        save_total_limit=2  # Keep only the last 2 checkpoints
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    
    eval_results = trainer.evaluate()
    
    torch.cuda.empty_cache()  # Clear memory after training
    
    return eval_results['eval_loss']

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

print("Best hyperparameters:", study.best_trial.params)
print("Best validation loss:", study.best_value)
