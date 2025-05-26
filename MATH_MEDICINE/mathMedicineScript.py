from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pickle

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LoRA_RANK = 4
LoRA_ALPHA = 64
LoRA_DROPOUT = 0.3
MAX_LENGTH = 64
BATCH_SIZE = 4
NUM_EPOCHS = 2

# Load and prepare datasets
medicine_dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")["train"]
math_dataset     = load_dataset("math_qa.py")["train"]

# Combine into unified lists
medicine = medicine_dataset.map(lambda x: {"question": x['Question'], "answer": x['Answer']})
math     = math_dataset.map(lambda x: {"question": x['Problem'],  "answer": x['Rationale']})
questions = math['question'] + medicine['question']
answers   = math['answer']   + medicine['answer']

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize data
tokenized_inputs = tokenizer(
    questions,
    truncation=True,
    padding='max_length',
    max_length=MAX_LENGTH,
    return_tensors="pt"
)
tokenized_targets = tokenizer(
    answers,
    truncation=True,
    padding='max_length',
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

# Custom dataset
def build_dataset(inputs, targets):
    class QADataset(Dataset):
        def __init__(self, inputs, targets):
            self.inputs  = inputs
            self.targets = targets
        def __len__(self):
            return len(self.inputs['input_ids'])
        def __getitem__(self, idx):
            return {
                'input_ids':      self.inputs['input_ids'][idx],
                'attention_mask': self.inputs['attention_mask'][idx],
                'labels':         self.targets['input_ids'][idx]
            }
    return QADataset(inputs, targets)

full_ds = build_dataset(tokenized_inputs, tokenized_targets)

total = len(full_ds)
train_size = int(0.5 * total)
val_size   = int(0.1 * total)
test_size  = total - train_size - val_size
train_ds, val_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, val_size, test_size])

# Configure LoRA
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

# Training arguments
training_args = TrainingArguments(
    output_dir="./math_medicine_lora",    
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=50
)

# Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

# Fine-tune LoRA
print("Starting fine-tuning on Math+Medicine data...")
trainer.train()

# Save the full LoRA-wrapped model
trainer.save_model("./math_medicine_lora")

# Extract and save LoRA weights only
# Reload base model and wrap with PeftModel to load checkpoint
peft_model = PeftModel.from_pretrained(base_model, "./math_medicine_lora").to(device)
lora_weights = {}
for name, param in peft_model.named_parameters():
    if 'lora_' in name:
        lora_weights[name] = param.detach().cpu().numpy()

with open("mathMedicine_lora_weights.pkl", "wb") as f:
    pickle.dump(lora_weights, f)

print(f"Saved LoRA weights ({len(lora_weights)} layers) to mathMedicine_lora_weights.pkl")

