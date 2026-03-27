"""
Compute Pointwise V-Information (PVI) metrics.

Adapted from: https://github.com/kawine/dataset_difficulty/blob/main/v_info.py

PVI measures how much product text reduces uncertainty about an LLM's
purchase decision. For each listing:

    PVI(x) = H_V(Y | empty) - H_V(Y | x)

where H_V is the V-entropy from Xu et al. (2020), computed using
fine-tuned classifiers as the predictive family V.

- H_V(Y | empty) is estimated by a "null model" trained on empty text.
- H_V(Y | x) is estimated by a "regular model" trained on actual text.

A positive PVI means the text is informative for the AI's decision;
the average PVI across post-ChatGPT listings vs. pre-ChatGPT listings
is the paper's main treatment effect.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftConfig, PeftModel
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path


def build_classifier(model_path: str, tokenizer, device):
    """Load a PEFT (LoRA) classifier and wrap it in a HuggingFace pipeline."""
    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base_id = peft_cfg.base_model_name_or_path
        base = AutoModelForSequenceClassification.from_pretrained(base_id, torch_dtype=torch.bfloat16)
        peft = PeftModel.from_pretrained(base, model_path)
        peft.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        peft.config.pad_token_id = tokenizer.pad_token_id
        return pipeline(
            "text-classification",
            model=peft,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=device,
        )


def v_entropy(data_fn, model, tokenizer, label_col, input_key='sentence1', batch_size=32):
    """
    Calculate the V-entropy (in bits) on the data given in data_fn. This can be
    used to calculate both the V-entropy and conditional V-entropy (for the
    former, the input column would only have null data and the model would be
    trained on this null data).

    Args:
        data_fn: path to data; should contain the label in the 'label' column
        model: path to saved model or model name in HuggingFace library
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        input_key: column name of X variable in data_fn
        batch_size: data batch_size

    Returns:
        Tuple of (V-entropies, correctness of predictions, predicted labels).
        Each is a List of n entries (n = number of examples in data_fn).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = build_classifier(model, tokenizer, device=torch.device(device))

    data = pd.read_json(data_fn, lines=True)

    entropies = []
    correct = []
    predicted_labels = []

    for j in tqdm(range(0, len(data), batch_size)):
        batch = data[j:j+batch_size]
        predictions = classifier(batch[input_key].tolist(), batch_size=batch_size)

        for i in range(len(batch)):
            gold = batch.iloc[i][label_col]
            target = f"LABEL_{int(gold)}"
            prob = next((d["score"] for d in predictions[i] if d["label"] == target), 1e-12)
            entropies.append(-np.log2(prob))
            pred_label = max(predictions[i], key=lambda x: x["score"])["label"]
            pred_idx = int(pred_label.split("_")[-1])
            predicted_labels.append(pred_idx)
            correct.append(pred_idx == int(gold))

    if device == "cuda":
        torch.cuda.empty_cache()

    return entropies, correct, predicted_labels


def v_info(data_fn, model, null_data_fn, null_model, tokenizer, label_col, out_fn="", input_key='sentence1'):
    """
    Calculate the V-entropy, conditional V-entropy, and V-information on the
    data in data_fn. Add these columns to the data in data_fn and return as a
    pandas DataFrame. This means that each row will contain the (pointwise
    V-entropy, pointwise conditional V-entropy, and pointwise V-info (PVI)). By
    taking the average over all the rows, you can get the V-entropy, conditional
    V-entropy, and V-info respectively.

    Args:
        data_fn: path to data; should contain the label in the 'label' column
            and X in column specified by input_key
        model: path to saved model or model name in HuggingFace library
        null_data: path to null data (column specified by input_key should have
            null data)
        null_model: path to saved model trained on null data
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        out_fn: where to saved
        input_key: column name of X variable in data_fn

    Returns:
        Pandas DataFrame of the data in data_fn, with the three additional
        columns specified above.
    """
    data = pd.read_json(data_fn, lines=True)
    data['H_yb'], _, _ = v_entropy(null_data_fn, null_model, tokenizer, label_col, input_key=input_key)
    data['H_yx'], data['correct_yx'], data['predicted_label'] = v_entropy(data_fn, model, tokenizer, label_col, input_key=input_key)
    data['PVI'] = data['H_yb'] - data['H_yx']

    if out_fn:
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)

        # Select only essential columns to reduce file size
        essential_columns = [
            'id',                # Merge key
            'H_yb', 'H_yx', 'PVI',  # V-entropy metrics
            'predicted_label', 'correct_yx',  # Model outputs
            'title', 'item_details',  # Source text for reference
        ]

        # Filter columns that exist in the data
        available_columns = [col for col in essential_columns if col in data.columns]
        original_size = len(data.columns)
        filtered_size = len(available_columns)
        print(f"Reduced columns from {original_size} to {filtered_size}")

        # Filter and save
        data_minimal = data[available_columns]
        data_minimal.to_json(out_fn, lines=True, orient='records')

    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute V-information metrics.")
    parser.add_argument('--data_dir', help='raw_data directory', required=True, type=str)
    parser.add_argument('--base_model_path', help='base model path', required=True, type=str)
    parser.add_argument('--label_col', help='label column name', required=True, type=str)
    parser.add_argument('--null_model_path', help='null model path (overrides default)', required=False, type=str, default=None)
    parser.add_argument('--split', choices=['test', 'validation', 'train'], default='test',
                        help='Dataset split to compute PVI on (default: test)')

    args = parser.parse_args()

    DATA_DIR = args.data_dir
    MODEL_DIR = DATA_DIR
    BASE_MODEL_PATH = args.base_model_path
    data_path = Path(DATA_DIR) / f"{args.split}_dataset.jsonl"
    if args.null_model_path:
        null_model_path = Path(args.null_model_path)
    else:
        null_model_path = Path(f"{MODEL_DIR}_null")
    null_data_path = null_model_path / f"{args.split}_dataset.jsonl"
    model_name = os.path.basename(MODEL_DIR)
    dir_name = model_name.split("/")[-1]

    if not data_path.is_file():
        print(f"[v_info] Skip: missing data file '{data_path}'.")
        raise SystemExit(0)
    if not null_data_path.is_file() or not null_model_path.exists():
        print(f"[v_info] Skip: missing null resources for '{MODEL_DIR}' "
              f"(expected '{null_data_path}' and '{null_model_path}').")
        raise SystemExit(0)

    tok = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    v_info(
        data_fn=str(data_path),
        model=str(MODEL_DIR),
        null_data_fn=str(null_data_path),
        null_model=str(null_model_path),
        tokenizer=tok,
        out_fn=str(Path(DATA_DIR).parent.parent / "results" / f"{dir_name}_PVI_{args.split}.json"),
        label_col=args.label_col,
        input_key='text'
    )
