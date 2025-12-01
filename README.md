# sumLoRA: Modular Domain Adaptation with LoRA

This project investigates whether LoRA adapters trained on different domains can be **combined or separated without retraining**.  
The core idea is that LoRA weight updates from independent datasets often lie in **approximately orthogonal subspaces**, making simple operations like delta addition surprisingly effective.

The repository includes code for training LoRA adapters, analyzing their structure with PCA, and composing multiple LoRAs into a single model.

---

## Features
- Train LoRA adapters on separate domains  
- Extract and analyze LoRA deltas  
- PCA visualization of delta subspaces  
- Naive LoRA summation (A + B)  
- Evaluate combined-domain performance  

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/LaziestCactus/sumLoRA.git
cd sumLoRA
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Training

Train a LoRA adapter on a dataset:

```bash
python training/train_lora.py --config training/config.yaml
```

This produces plots showing the principal directions and subspace overlaps.

---

## Composing LoRA Adapters

(Coming soon or add your script here.)

---

## Citation

If you use this project, please cite:

```
Cao, Z. (2025). Efficient Modular Learning through Naive LoRA Summation.
arXiv:2508.11985.
```

