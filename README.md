# Mecha-nudges for Machines

Code companion for the paper *"Mecha-nudges for Machines"*.

## Abstract

Nudges are subtle changes to the way choices are presented to human decision-makers (e.g., opt-in vs. opt-out by default) that shift behavior without restricting options or changing incentives. As AI agents increasingly make decisions in the same environments as humans, the presentation of choices may be optimized for machines as well as people. We introduce mecha-nudges: changes to how choices are presented that systematically influence AI agents without degrading the decision environment for humans. To formalize mecha-nudges, we combine the Bayesian persuasion framework with V-usable information, a generalization of Shannon information that is observer-relative. This yields a common scale (bits of usable information) for comparing a wide range of interventions, contexts, and models. Applying our framework to product listings on Etsy -- a global marketplace for independent sellers -- we find that following ChatGPT's release, listings have significantly more machine-usable information about product selection, consistent with systematic mecha-nudging.

## Overview

**Core metric:** Pointwise V-Information (PVI) measures how much product text reduces uncertainty about an LLM's purchase decision:

```
PVI(x -> y) = -log2 P(y | empty) + log2 P(y | x)
```

where `y` is the gold label (SELECT/PASS) and `P` comes from fine-tuned classifiers. PVI is evaluated at each listing's gold label, so it depends on both the text `x` and the label `y`.

**Key finding:** Post-ChatGPT listings contain 0.143 additional bits of machine-usable information (p < 0.01).

## Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   create_y.py   │     │balance_labels.py │     │  finetuning.py   │     │    v_info.py    │
│                 │     │                  │     │                  │     │                 │
│  LLM labels     │────>│  Undersample to  │────>│  LoRA classifier │────>│  PVI = H(Y|∅)   │
│  (SELECT/PASS)  │     │  equal classes   │     │  + null model    │     │       - H(Y|x)  │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └─────────────────┘
```

1. **Label Generation** (`src/create_y.py`): Prompt an LLM with product info and force a binary SELECT/PASS decision via constrained generation. A custom `PosNegLogitsProcessor` masks all vocabulary tokens except the two decision tokens, guaranteeing a valid binary output with calibrated probabilities. The paper's main results use GPT-5-mini (a closed model); the code defaults to Gemma-3-27B-IT for reproducibility with open-weights models.

2. **Class Balancing** (`src/balance_labels.py`): Undersample the majority class to create equal positive/negative representation. Raw label distributions are highly imbalanced (~16% positive pre-ChatGPT, ~5% post-ChatGPT with GPT-5-mini); balancing ensures fine-tuned models learn the content–label relationship rather than just the base rate.

3. **Fine-Tuning** (`src/finetuning.py`): LoRA fine-tune a classifier (Llama-3.1-8B-Instruct) on the balanced labels. A second "null" model trained on empty text establishes the baseline entropy H(Y|empty). Separate model pairs are trained for each time period.

4. **V-Information** (`src/v_info.py`): Compute PVI = H(Y|empty) - H(Y|text) for each listing using the fine-tuned models.

5. **Regression** (`src/regressions.py`): OLS regression of PVI on a post-ChatGPT indicator.

**Supporting files:**

- `src/generation_utils.py`: Core utilities shared across the pipeline — `prompt_creator` (builds the LLM prompt from product attributes), `PosNegLogitsProcessor` (constrained decoding to force a binary token output), and `preprocess_variables_list` (normalizes input fields like `item_details`).
- `prompts/`: The four prompt variants used in the paper's robustness analysis (`prompt_v1.txt`–`prompt_v4.txt`). V4 is the primary prompt; V1–V3 are alternatives tested in the robustness table.

## Requirements

- Python 3.12
- CUDA 12.4 (for GPU-accelerated steps: label generation, fine-tuning, V-information)

## Reproducing the main result

```bash
pip install -r requirements.txt

python src/regressions.py --before data/before_PVI.json --after data/after_PVI.json
```

This runs `PVI ~ after` on the included PVI data (SELECT/PASS tokens, balanced medium dataset, labeled with GPT-5-mini in the paper) and should output a coefficient of **0.143** (p < 0.01).

## Running the full pipeline

Each step can be run independently on your own data:

```bash
# Step 1: Generate labels
python src/create_y.py --input data/etsy.jsonl --output out/labels.jsonl \
    --input-length 100000 --variable Title=title Item_Details=item_details \
    --positive-token SELECT --negative-token PASS \
    --prompt-template-file prompts/prompt_v4.txt --batch-size 8

# Step 2: Balance labels (undersample majority class)
python src/balance_labels.py --input out/labels.jsonl \
    --output out/labels_balanced.jsonl --label-col y_title_item_details

# Step 3: Fine-tune classifier (regular + null models, per period)
python src/finetuning.py --dataset-path out/labels_balanced.jsonl \
    --label-column y_title_item_details --output-dir out/model \
    --positive-token SELECT --negative-token PASS --use-stratified-split

python src/finetuning.py --dataset-path out/labels_balanced.jsonl \
    --label-column y_title_item_details --output-dir out/model_null \
    --null-model --use-stratified-split

# Step 4: Compute PVI
python src/v_info.py --data-dir out/model --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --label-col y_title_item_details
```

## Robustness

The paper tests robustness across several dimensions:

| Dimension | Primary | Alternatives |
|-----------|---------|-------------|
| Token pair | SELECT/PASS | YES/NO, BUY/SKIP, PICK/PASS |
| Prompt | V4 (descriptive) | V1 (minimal), V2 (clarified), V3 (indirect) |
| Labeling model | GPT-5-mini | Gemma-3-27B-IT, Qwen3-32B |
| Fine-tuning model | Llama-3.1-8B-Instruct | Qwen3-8B, Gemma-3-12B |
| Seed | 42 | 123, 456, 789, 1024 |

The paper also reports the following analyses. They are described there, but their code is not included in this repository:

- **DailyMed placebo**: The same pipeline is applied to pharmaceutical drug labels (which should be unaffected by ChatGPT) as a temporal control.
- **Rephrasing placebo**: Pre-ChatGPT listings are rephrased with GPT-5-mini to test whether the PVI increase reflects general LLM-style text rather than mecha-nudging.
- **Token-level ablation**: Individual words are removed from listings and PVI is recomputed to identify which tokens drive the effect.

## Citation

```bibtex
@misc{mechanudges2026,
  title         = {Mecha-nudges for Machines},
  author        = {Giulio Frey and Kawin Ethayarajh},
  year          = {2026},
  eprint        = {2603.23433},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

