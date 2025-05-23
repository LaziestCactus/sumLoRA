from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Dataset
import optuna
from datasets import load_dataset
import os
import json

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

medicine_dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
math_dataset = load_dataset("math_qa.py")

medicine_train = medicine_dataset['train']
math_train = math_dataset['train']

medicine_combined = medicine_train.map(lambda x: {
    'question': x['Question'],
    'answer': x['Answer'],
    'category': 'medicine'  
})

math_combined = math_train.map(lambda x: {
    'question': x['Problem'],  
    'answer': x['Rationale'], 
    'category': 'math',  
})

def combine_data(math_data, medical_data):
    questions = math_data["question"] + medical_data["question"]
    answers = math_data["answer"] + medical_data["answer"]
    categories = math_data["category"] + medical_data["category"]
    return questions, answers, categories

questions, answers, categories = combine_data(math_combined, medicine_combined)

# Custom Dataset Class
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

# Load model and tokenizer
model_name = "gpt2"
GPTmodel = GPT2LMHeadModel.from_pretrained(model_name).to(device)
GPTmodel.gradient_checkpointing_enable()  # Reduce memory consumption
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Tokenize dataset
tokenizer.pad_token = tokenizer.eos_token
tokenized_questions = tokenizer(questions, truncation=True, padding='max_length', max_length=128, return_tensors="np")
tokenized_answers = tokenizer(answers, truncation=True, padding='max_length', max_length=128, return_tensors="np")

# Create dataset
full_dataset = makeDataset(tokenized_questions, tokenized_answers)

# Split dataset
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])


# Hyperparameters to tune
LoRA_Rank = 8
LoRA_Alpha = 32 
LoRA_Dropout = 0.3
batch_size = 4
num_epochs = 2

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LoRA_Rank,
    lora_alpha=LoRA_Alpha,
    lora_dropout=LoRA_Dropout,
    target_modules=["c_attn", "c_proj"],
    modules_to_save=["lm_head"]
)

# Prepare model with LoRA
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
    save_strategy="epoch",  # Save at the end of each epoch
    save_total_limit=2,  # Keep the latest 2 checkpoints to save space
    load_best_model_at_end=True  # Load the best checkpoint after training
)   

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()
