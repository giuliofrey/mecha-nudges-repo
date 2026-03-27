# Mecha-nudges for Machines

Code companion for the paper *"Mecha-nudges for Machines"*.

## Abstract

Nudges are subtle changes to the way choices are presented to human decision-makers (e.g., opt-in vs. opt-out by default) that shift behavior without restricting options or changing incentives. As AI agents increasingly make decisions in the same environments as humans, the presentation of choices may be optimized for machines as well as people. We introduce mecha-nudges: changes to how choices are presented that systematically influence AI agents without degrading the decision environment for humans. To formalize mecha-nudges, we combine the Bayesian persuasion framework with V-usable information, a generalization of Shannon information that is observer-relative. This yields a common scale (bits of usable information) for comparing a wide range of interventions, contexts, and models. Applying our framework to product listings on Etsy -- a global marketplace for independent sellers -- we find that following ChatGPT's release, listings have significantly more machine-usable information about product selection, consistent with systematic mecha-nudging.

## Overview

**Core metric:** Pointwise V-Information (PVI) measures how much product text reduces uncertainty about an LLM's purchase decision:

```
PVI(x) = H_V(Y | empty) - H_V(Y | x)
```

where `H_V` is the V-entropy (in bits) from Xu et al. (2020), computed using fine-tuned classifiers.

**Key finding:** Post-ChatGPT listings contain 0.143 additional bits of machine-usable information (p < 0.01).

## Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   create_y.py   │     │  finetuning.py   │     │    v_info.py    │
│                 │     │                  │     │                 │
│  LLM labels     │────>│  LoRA classifier │────>│  PVI = H(Y|∅)  │
│  (SELECT/PASS)  │     │  + null model    │     │       - H(Y|x)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

1. **Label Generation** (`src/create_y.py`): Prompt an LLM (Gemma-3-27B) with product info and force a binary SELECT/PASS decision via constrained generation. A custom `PosNegLogitsProcessor` masks all vocabulary tokens except the two decision tokens, guaranteeing a valid binary output with calibrated probabilities.

2. **Fine-Tuning** (`src/finetuning.py`): LoRA fine-tune a classifier (Llama-3.1-8B) on the labels. A second "null" model trained on empty text establishes the baseline entropy H(Y|empty).

3. **V-Information** (`src/v_info.py`): Compute PVI = H(Y|empty) - H(Y|text) for each listing using the two fine-tuned models.

4. **Regression** (`src/regressions.py`): OLS regression of PVI on a post-ChatGPT indicator.

## Reproducing the main result

```bash
pip install -r requirements.txt

python src/regressions.py --before data/before_PVI.json --after data/after_PVI.json
```

This runs `PVI ~ after` on the included PVI data (SELECT/PASS tokens, GPT-5-mini labels, balanced medium dataset) and should output a coefficient of **0.143** (p < 0.01).

## Running the full pipeline

Each step can be run independently on your own data:

```bash
# Step 1: Generate labels
python src/create_y.py --input data/etsy.jsonl --output out/labels.jsonl \
    --input-length 100000 --variable Title=title Item_Details=item_details \
    --positive-token SELECT --negative-token PASS \
    --prompt-template-file prompts/prompt_v4.txt --batch-size 8

# Step 2: Fine-tune classifier (regular + null models)
python src/finetuning.py --dataset-path out/labels.jsonl \
    --label-column y_title_item_details --output-dir out/model \
    --positive-token SELECT --negative-token PASS --use-stratified-split

python src/finetuning.py --dataset-path out/labels.jsonl \
    --label-column y_title_item_details --output-dir out/model_null \
    --null-model --use-stratified-split

# Step 3: Compute PVI
python src/v_info.py --data_dir out/model --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --label_col y_title_item_details
```

## Robustness

The paper tests robustness across several dimensions:

| Dimension | Primary | Alternatives |
|-----------|---------|-------------|
| Token pair | SELECT/PASS | YES/NO, BUY/SKIP, PICK/PASS |
| Prompt | V4 (descriptive) | V1 (minimal), V2 (clarified), V3 (indirect) |
| Labeling model | GPT-5-mini | Gemma-3-27B, Qwen3-32B |
| Fine-tuning model | Llama-3.1-8B | Qwen-2.5, Gemma |
| Seed | 42 | 123, 456, 789, 1024 |

## Citation

```bibtex
@article{mechanudges,
  title={Mecha-nudges for Machines},
  author={Frey, Giulio and Ethayarajh, Kawin},
  year={2026},
  url={https://arxiv.org/abs/2603.23433}
}
```

## License

See [LICENSE](LICENSE) for details.
