from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.amp import autocast
from datasets import load_dataset
import matplotlib as plt
import os
import optuna

# Load the dataset
#https://huggingface.co/datasets/allenai/math_qa
import datasets

ds = load_dataset("math_qa.py")
# Access the different splits
train_data = ds['train']
validation_data = ds['validation']
test_data = ds['test']

# Access the first training example
#print(train_data[0])
# Combine corresponding elements of "Problem" and "options"
training_question = list([p + " " + o for p, o in zip(train_data[:]["Problem"], train_data[:]["options"])])
training_answer = list(train_data[:]['Rationale'])
validation_question = list([p + " " + o for p, o in zip(validation_data[:]["Problem"], train_data[:]["options"])])
validation_answer = list(validation_data[:]['Rationale'])
test_question = list([p + " " + o for p, o in zip(test_data[:]["Problem"], train_data[:]["options"])])
test_answer = list(test_data[:]['Rationale'])
print(training_question[0])
print(validation_answer[134])
print(test_answer[134])


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



#HYPERPARAMS:
LoRA_Rank = 4
LoRA_Alpha = 64
LoRA_Dropout = 0.1
max_length = 64 #longest token taken
batch_size = 16
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
    target_modules = ["c_attn", "c_proj"]
)

# Prepare model for LoRA tuning
model = get_peft_model(GPTmodel, lora_config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

#tokenize dataset
tokenizer.pad_token = tokenizer.eos_token
tokenized_training_question = tokenizer(training_question, truncation=True, padding='max_length', return_tensors="pt", max_length = max_length)
tokenized_training_answer = tokenizer(training_answer, truncation=True, padding=True, return_tensors="pt", max_length = max_length)
tokenized_validation_question = tokenizer(validation_question, truncation=True, padding='max_length', return_tensors="pt", max_length = max_length)
tokenized_validation_answer = tokenizer(validation_answer, truncation=True, padding=True, return_tensors="pt", max_length = max_length)
tokenized_test_question = tokenizer(test_question, truncation=True, padding='max_length', return_tensors="pt", max_length = max_length)
tokenized_test_answer = tokenizer(test_answer, truncation=True, padding=True, return_tensors="pt", max_length = max_length)

print(f"Tokenized Training Questions Shape: {tokenized_training_question['input_ids'].shape}")
print(f"Tokenized Training Answers Shape: {tokenized_training_answer['input_ids'].shape}")


# Make sure it's divisible by batch size so last batch works fine
train_dataset = makeDataset(tokenized_training_question, tokenized_training_answer)
val_dataset = makeDataset(tokenized_validation_question, tokenized_validation_answer)
test_dataset = makeDataset(tokenized_test_question, tokenized_test_answer)
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',        # Directory to save model checkpoints
    num_train_epochs=num_epochs,            # Number of training epochs
    per_device_train_batch_size=batch_size, # Batch size per device
    per_device_eval_batch_size=batch_size,  # Batch size for evaluation
    warmup_steps=2,              # Number of warmup steps
    weight_decay=0.01,             # Weight decay
    logging_dir='./logs',          # Directory to save logs
    logging_steps=50,              # Log every X steps
)

# Create Trainer instance
trainer = Trainer(
    model=model,                     # The model you are fine-tuning
    args=training_args,              # Training arguments
    train_dataset=train_dataset,     # Your training dataset
    eval_dataset=val_dataset,
)

# Get model sizes
def print_model_size(path):
    size = 0
    for f in os.scandir(path):
        size += os.path.getsize(f)
    print(f"Model size: {(size / 1e6):.2} MB")

def print_trainable_parameters(model, label):
    parameters, trainable = 0, 0    
    for _, p in model.named_parameters():
        parameters += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    print(f"{label} trainable parameters: {trainable:,}/{parameters:,} ({100 * trainable / parameters:.2f}%)")

#Fine-tune the model
print(f"Model is on device: {next(model.parameters()).device}")
print_model_size(training_args.output_dir)
print_trainable_parameters(model, "Before training")


#Training the model
trainer.train()
trainer.save_model("./mathLora-model")
print("_________________________________")
print("completed running default lora")


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
model_name = "mathLora-model"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Evaluate the model
model.eval()
total_loss = 0
num_batches = 0
batch_size = 8
with torch.no_grad():
    for i in range(0, len(val_dataset), batch_size):
        if(i+batch_size >= len(val_dataset)):
            break
        batch = val_dataset[i:i + batch_size]
        # Get input_ids and attention_mask from the batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask'] if 'attention_mask' in batch else None  # Optional

        # Pass input_ids as labels for loss calculation
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels= batch['labels'])
        
        loss = outputs.loss
        total_loss += loss.item()
        num_batches += 1

# Calculate average loss and perplexity
average_loss = total_loss / num_batches
perplexity = torch.exp(torch.tensor(average_loss)).item()

print(f"Average Loss: {average_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")



import pickle
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
# Load the base GPT-2 model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load the fine-tuned model with LoRA layers
model = PeftModel.from_pretrained(base_model, "mathLora-model")

# Initialize a dictionary to store LoRA weights
lora_weights = {}

# Iterate through model parameters and extract LoRA layers
count = 0
for name, param in model.named_parameters():
    if 'lora' in name:
        count = count + 1
        print(f"Extracting LoRA Layer: {name}, Shape: {param.shape}")
        lora_weights[name] = param.detach().cpu().numpy()
        
print(f"The number of modified parameter is {count}.")

# Save the LoRA weights to a file
with open("math_lora_weights.pkl", "wb") as f:
    pickle.dump(lora_weights, f)

print("LoRA weights extracted and saved to lora_weights.pkl")




