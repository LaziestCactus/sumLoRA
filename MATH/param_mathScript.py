from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
import os
import optuna
import datasets

# Load the dataset
ds = datasets.load_dataset("math_qa.py")
# Access the different splits
train_data = ds['train']
validation_data = ds['validation']
test_data = ds['test']

# Combine corresponding elements of "Problem" and "options"
training_question = list([p + " " + o for p, o in zip(train_data[:]["Problem"], train_data[:]["options"])])
training_answer = list(train_data[:]['Rationale'])
validation_question = list([p + " " + o for p, o in zip(validation_data[:]["Problem"], train_data[:]["options"])])
validation_answer = list(validation_data[:]['Rationale'])
test_question = list([p + " " + o for p, o in zip(test_data[:]["Problem"], train_data[:]["options"])])
test_answer = list(test_data[:]['Rationale'])

class makeDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        # Extract the input_ids and attention_mask for the question
        input_ids = self.inputs['input_ids'][idx]
        attention_mask = self.inputs['attention_mask'][idx]

        # Extract the labels (input_ids for the answer)
        labels = self.targets['input_ids'][idx]

        # Return the input and output as a dictionary
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# HYPERPARAMS:
LoRA_Rank = 8
LoRA_Alpha = 32
LoRA_Dropout = 0.1
max_length = 64  # longest token taken
batch_size = 32
num_epochs = 1

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # Default to GPT small
GPTmodel = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LoRA_Rank,  # Rank of the low-rank adaptation matrices
    lora_alpha=LoRA_Alpha,  # LoRA scaling factor
    lora_dropout=LoRA_Dropout,  # Dropout for LoRA layers
    target_modules=["c_attn", "c_proj"]
)

# Prepare model for LoRA tuning
model = get_peft_model(GPTmodel, lora_config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Tokenize dataset
tokenizer.pad_token = tokenizer.eos_token
tokenized_training_question = tokenizer(training_question, truncation=True, padding='max_length', return_tensors="pt", max_length=max_length)
tokenized_training_answer = tokenizer(training_answer, truncation=True, padding=True, return_tensors="pt", max_length=max_length)
tokenized_validation_question = tokenizer(validation_question, truncation=True, padding='max_length', return_tensors="pt", max_length=max_length)
tokenized_validation_answer = tokenizer(validation_answer, truncation=True, padding=True, return_tensors="pt", max_length=max_length)
tokenized_test_question = tokenizer(test_question, truncation=True, padding='max_length', return_tensors="pt", max_length=max_length)
tokenized_test_answer = tokenizer(test_answer, truncation=True, padding=True, return_tensors="pt", max_length=max_length)

# Make sure it's divisible by batch size so last batch works fine
train_dataset = makeDataset(tokenized_training_question, tokenized_training_answer)
val_dataset = makeDataset(tokenized_validation_question, tokenized_validation_answer)
test_dataset = makeDataset(tokenized_test_question, tokenized_test_answer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',        # Directory to save model checkpoints
    num_train_epochs=num_epochs,   # Number of training epochs
    per_device_train_batch_size=batch_size,  # Batch size per device
    per_device_eval_batch_size=batch_size,  # Batch size for evaluation
    warmup_steps=2,                # Number of warmup steps
    weight_decay=0.01,             # Weight decay
    logging_dir='./logs',          # Directory to save logs
    logging_steps=50,              # Log every X steps
)

# Create Trainer instance
trainer = Trainer(
    model=model,                    # The model you are fine-tuning
    args=training_args,             # Training arguments
    train_dataset=train_dataset,    # Your training dataset
    eval_dataset=val_dataset,
)

# Fine-tune the model
# trainer.train()

def objective(trial):
    # Sample hyperparameters from the search space
    LoRA_Rank = trial.suggest_categorical('LoRA_Rank', [4, 8, 16])
    LoRA_Alpha = trial.suggest_categorical('LoRA_Alpha', [16, 32, 64])
    LoRA_Dropout = trial.suggest_float('LoRA_Dropout', 0.1, 0.3, step=0.05)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_epochs = trial.suggest_int('num_epochs', 1, 3)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LoRA_Rank,
        lora_alpha=LoRA_Alpha,
        lora_dropout=LoRA_Dropout,
        target_modules = ["c_attn", "c_proj"]
    )

    # Re-initialize the model for each trial
    model = get_peft_model(GPTmodel, lora_config).to(device)

    # Create a custom DataLoader that moves tensors to the correct device
    class DeviceAwareDataLoader(DataLoader):
        def __init__(self, dataset, device, **kwargs):
            super().__init__(dataset, **kwargs)
            self.device = device

        def __iter__(self):
            iterator = super().__iter__()
            for batch in iterator:
                yield {k: v.to(self.device) for k, v in batch.items()}

    # Create dataloaders with device awareness
    train_loader = DeviceAwareDataLoader(
        train_dataset,
        device=device,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DeviceAwareDataLoader(
        val_dataset,
        device=device,
        batch_size=batch_size,
        shuffle=False
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_trial_{trial.number}',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir='./logs',
        logging_steps=50
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the validation set
    total_loss = 0
    num_batches = 0
    model.eval()

    with torch.no_grad():
        for batch in val_loader:  # Use the device-aware loader
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'] if 'attention_mask' in batch else None,
                labels=batch['labels']
            )
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

    # Calculate average loss
    average_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    return average_loss

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize')  # Minimize loss
study.optimize(objective, n_trials=10)

# Print the best trial's hyperparameters
print(f"Best hyperparameters: {study.best_trial.params}")
print(f"Best validation loss: {study.best_value}")
