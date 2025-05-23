from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.amp import autocast
from datasets import load_dataset
import matplotlib as plt
import matplotlib.pyplot as plt
import os

print("Hello World")
