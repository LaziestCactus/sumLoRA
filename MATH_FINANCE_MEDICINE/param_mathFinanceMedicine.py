import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
import optuna

# ─── Load Datasets ─────────────────────────────────────────────────────
financial_train = load_dataset("itzme091/financial-qa-10K-modified")['train']
medicine_train = load_dataset("keivalya/MedQuad-MedicalQnADataset")['train']
math_train = load_dataset("math_qa.py")['train']

# ─── Format Datasets ────────────────────────────────────────────────────
financial_combined = financial_train.map(lambda x: {
    'question': x['question'],
    'answer': x['answer'],
    'category': 'financial'
})
medicine_combined = medicine_train.map(lambda x: {
    'question': x['Question'],
    'answer': x['Answer'],
    'category': 'medicine'
})
math_combined = math_train.map(lambda x: {
    'question': x['Problem'],
    'answer': x['Rationale'],
    'category': 'math'
})

# ─── Combine Datasets ───────────────────────────────────────────────────
def combine_data(financial_data, math_data, medicine_data):
    questions = financial_data["question"] + math_data["question"] + medicine_data["question"]
    answers = financial_data["answer"] + math_data["answer"] + medicine_data["answer"]
    categories = financial_data["category"] + math_data["category"] + medicine_data["category"]
    return questions, answers, categories

questions, answers, categories = combine_data(financial_combined, math_combined, medicine_combined)

# ─── Tokenizer & Device ────────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have pad_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Tokenize ──────────────────────────────────────────────────────────
inputs = tokenizer(questions, padding=True, truncation=True, max_length=128, return_tensors="pt")
targets = tokenizer(answers, padding=True, truncation=True, max_length=128, return_tensors="pt")

# ─── Dataset Class ─────────────────────────────────────────────────────
class makeDataset(Dataset):
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

# ─── Objective Function for Optuna ─────────────────────────────────────
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    num_epochs = trial.suggest_int("num_epochs", 1, 3)

    # Split data manually for demo purposes (e.g., 90% train / 10% val)
    split_idx = int(0.9 * len(inputs['input_ids']))
    train_dataset = makeDataset(
        {k: v[:split_idx] for k, v in inputs.items()},
        {k: v[:split_idx] for k, v in targets.items()}
    )
    val_dataset = makeDataset(
        {k: v[split_idx:] for k, v in inputs.items()},
        {k: v[split_idx:] for k, v in targets.items()}
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.resize_token_embeddings(len(tokenizer))  # pad_token added

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
        save_strategy="epoch",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results['eval_loss']

# ─── Run Optuna ────────────────────────────────────────────────────────
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

print("Best hyperparameters:", study.best_trial.params)
print("Best validation loss:", study.best_value)

