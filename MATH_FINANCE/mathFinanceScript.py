from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pickle

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LoRA_RANK    = 4
LoRA_ALPHA   = 64
LoRA_DROPOUT = 0.25
MAX_LENGTH   = 128
BATCH_SIZE   = 16 
NUM_EPOCHS   = 3

# Load datasets
finance_ds = load_dataset("itzme091/financial-qa-10K-modified")["train"]
math_ds    = load_dataset("math_qa.py")["train"]

# Extract question/answer pairs
finance = finance_ds.map(lambda x: {"question": x['question'], "answer": x['answer']})
math    = math_ds.map(lambda x:    {"question": x['Problem'],  "answer": x['Rationale']})

questions = finance['question'] + math['question']
answers   = finance['answer']   + math['answer']

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize inputs & targets
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

# Dataset wrapper
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
train_size = int(0.8 * total)
val_size   = int(0.1 * total)
test_size  = total - train_size - val_size
train_ds, val_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, val_size, test_size])

# LoRA configuration
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
    output_dir="./math_finance_lora",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=50
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

# Fine-tune LoRA on Math+Finance
torch.cuda.empty_cache()
print("Starting LoRA fine-tuning on Math & Finance...")
trainer.train()

# Save the full LoRA-wrapped model
trainer.save_model("./math_finance_lora")

# Extract and save only LoRA adapter weights
peft_model = PeftModel.from_pretrained(base_model, "./math_finance_lora").to(device)
lora_weights = {}
for name, param in peft_model.named_parameters():
    if name.startswith('lora_'):
        lora_weights[name] = param.detach().cpu().numpy()

with open("mathFinance_lora_weights.pkl", "wb") as f:
    pickle.dump(lora_weights, f)

print(f"Saved {len(lora_weights)} LoRA layers to mathFinance_lora_weights.pkl")

