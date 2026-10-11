# sumLoRA

Code for **[Efficient Modular Learning through Naive LoRA Summation: Leveraging Orthogonality in High-Dimensional Models](https://arxiv.org/abs/2508.11985)** (Cao, Truong, Lizarraga, 2025).

We train LoRA adapters on separate domains and combine them by adding their weight deltas, with no further training. Adapters trained on unrelated data tend to sit in nearly orthogonal subspaces, so they interfere very little when added.

## Results

The base model is GPT-2 Small (117M), with LoRA rank 4 and alpha 64. We train on three QA domains: math, medicine, and finance.

| Combination | Perplexity vs. training on merged data |
|---|---|
| Math + Medicine | **−9.10%** |
| Math + Finance | +4.54% |
| Finance + Medicine | +27.56% |

The RMS cosine similarity between two adapters' deltas predicts how much they interfere. The relationship is close to linear: the more similar the deltas, the bigger the perplexity penalty.

## Repository layout

There is one folder per domain (`MATH`, `MEDICINE`, `FINANCE`) and one per combination (`MATH_MEDICINE`, `MATH_FINANCE`, `FINANCE_MEDICINE`, `MATH_FINANCE_MEDICINE`). Each folder uses the same script naming:

| Script | Purpose |
|---|---|
| `param_*.py` | Optuna hyperparameter search for the LoRA config |
| `*Script.py` | Fine-tunes a LoRA adapter on that domain's data (or on merged data, as the baseline) |
| `create_*Weights.py` | Adds the per-domain LoRA deltas together |
| `lora_*Script.py` | Loads GPT-2 with the summed deltas and evaluates perplexity |

The best hyperparameters for each run are recorded in [`parameters.md`](parameters.md).

## Setup

```bash
git clone https://github.com/LaziestCactus/sumLoRA.git
cd sumLoRA
pip install -r requirements.txt
```

## Citation

```bibtex
@article{cao2025sumlora,
  title   = {Efficient Modular Learning through Naive LoRA Summation: Leveraging Orthogonality in High-Dimensional Models},
  author  = {Cao, Zhanhao and Truong, Clement and Lizarraga, Andrew},
  journal = {arXiv preprint arXiv:2508.11985},
  year    = {2025}
}
```
