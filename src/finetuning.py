"""
Step 2: LoRA Fine-Tuning for Binary Sequence Classification.

Fine-tunes a causal LM (default: Llama-3.1-8B-Instruct) as a binary classifier
using LoRA adapters. The pipeline trains two models:

  1. A **regular model** on the actual product text (measures H(Y|x)).
  2. A **null model** on empty text (measures H(Y|empty), the baseline entropy).

The difference between these two entropies is the Pointwise V-Information (PVI).

Usage:
    # Train regular model
    python finetuning.py \\
        --dataset-path out/y_labels/before_labels.jsonl \\
        --label-column y_title_item_details \\
        --output-dir out/models/before \\
        --positive-token SELECT --negative-token PASS

    # Train null model (baseline with empty text)
    python finetuning.py \\
        --dataset-path out/y_labels/before_labels.jsonl \\
        --label-column y_title_item_details \\
        --output-dir out/models/before_null \\
        --null-model
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import torch
from datasets import DatasetDict, load_dataset, Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from generation_utils import prompt_creator


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE = 12
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 3
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
MAX_LENGTH = 1024
SEED = 42
GRADIENT_ACCUMULATION_STEPS = 16
SAVE_TOTAL_LIMIT = 1

# Default base model — override via --base-model-path CLI argument
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_and_prepare_datasets(
    dataset_path: str,
    label_column: str,
    positive_token: str = "SELECT",
    negative_token: str = "PASS",
    prompt_variable: Optional[List[str]] = None,
    use_stratified_split: bool = False,
    seed: int = SEED,
) -> DatasetDict:
    """
    Load dataset from JSONL file and prepare it for training.

    Args:
        dataset_path: Path to JSONL file (output of create_y.py)
        label_column: Name of label column (e.g., "y_title_item_details")
        positive_token: Token representing positive class (default "SELECT")
        negative_token: Token representing negative class (default "PASS")
        prompt_variable: List of variable specifications from payload to use for prompt.
                        Supports mapping syntax: "prompt_key=payload_key" or just "key_name".
        use_stratified_split: Whether to use stratified splitting (default False)
        seed: Random seed for train/val/test splitting (default 42)

    Returns:
        DatasetDict with train, validation, and test splits
    """
    print(f"\nLoading dataset from {dataset_path}...")

    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    full_dataset = load_dataset(
        "json",
        data_files=str(dataset_path),
        split="train",
        keep_in_memory=True,
    )

    print(f"Total records loaded: {len(full_dataset)}")

    # Reconstruct prompts from stored payload
    def reconstruct_prompt(example):
        payload_cols = example.get("payload")
        if not isinstance(payload_cols, dict):
            return {"text": "", "label": None}

        # If specific prompt variable(s) are specified, use only those variables
        if prompt_variable:
            prompt_payload = {}
            for spec in prompt_variable:
                if '=' in spec:
                    prompt_key, payload_key = spec.split('=', 1)
                    if payload_key in payload_cols:
                        prompt_payload[prompt_key] = payload_cols[payload_key]
                else:
                    if spec in payload_cols:
                        prompt_payload[spec] = payload_cols[spec]
        else:
            # Use the full payload and handle item_details normalization
            prompt_payload = payload_cols.copy()
            if 'item_details' in prompt_payload:
                if isinstance(prompt_payload['item_details'], (list, tuple)):
                    prompt_payload['item_details'] = prompt_payload['item_details'][0] if prompt_payload['item_details'] else ""
                elif isinstance(prompt_payload['item_details'], str):
                    if prompt_payload['item_details'].strip().startswith('['):
                        try:
                            import ast
                            parsed = ast.literal_eval(prompt_payload['item_details'])
                            if isinstance(parsed, (list, tuple)):
                                prompt_payload['item_details'] = parsed[0] if parsed else ""
                        except (ValueError, SyntaxError):
                            pass

        prompt, prompt_details_only = prompt_creator(
            prompt_payload,
            positive_token=positive_token,
            negative_token=negative_token
        )

        # Extract label (Boolean) and convert to string
        label_value = example.get(label_column)
        label_str = str(label_value) if label_value is not None else None

        return {
            "text": prompt_details_only,
            "label": label_str,
        }

    print("Reconstructing prompts and extracting labels...")
    full_dataset = full_dataset.map(reconstruct_prompt)

    # Filter out null labels
    original_size = len(full_dataset)
    full_dataset = full_dataset.filter(lambda ex: ex["label"] is not None)
    full_dataset = full_dataset.filter(lambda ex: str(ex["label"]).strip() != "")

    if len(full_dataset) < original_size:
        print(f"Filtered out {original_size - len(full_dataset)} records with null/empty labels")

    print(f"Records after filtering: {len(full_dataset)}")

    # Log label distribution
    label_counts = {}
    for example in full_dataset:
        label = example["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(full_dataset)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    # Split into train, validation, test
    if use_stratified_split:
        print("\nSplitting into train/val/test with stratification...")
        from datasets import ClassLabel
        unique_labels = sorted(full_dataset.unique("label"))
        class_label = ClassLabel(names=[str(label) for label in unique_labels])
        full_dataset = full_dataset.cast_column("label", class_label)

        train_test = full_dataset.train_test_split(
            test_size=0.2, seed=seed, stratify_by_column="label"
        )
        val_test = train_test["test"].train_test_split(
            test_size=0.5, seed=seed, stratify_by_column="label"
        )
    else:
        print("\nSplitting into train/val/test (no stratification)...")
        train_test = full_dataset.train_test_split(test_size=0.2, seed=seed)
        val_test = train_test["test"].train_test_split(test_size=0.5, seed=seed)

    datasets = DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

    print(f"  Train: {len(datasets['train'])} samples")
    print(f"  Validation: {len(datasets['validation'])} samples")
    print(f"  Test: {len(datasets['test'])} samples")

    return datasets


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(datasets: DatasetDict, label_column: str = "label") -> Tuple[DatasetDict, Dict[str, int], str]:
    """
    Encode string labels to integer IDs for classification.
    Handles Boolean string representations ("True"/"False").
    """
    print(f"\nEncoding labels from column '{label_column}'...")

    label_values = {
        label
        for ds in datasets.values()
        for label in ds.unique(label_column)
    }

    if len(label_values) != 2:
        raise ValueError(
            f"Expected 2 unique labels in '{label_column}', found {len(label_values)}: {label_values}"
        )

    # Sort labels: False (negative) -> 0, True (positive) -> 1
    sort_key = lambda value: (value is None, str(value))
    label_list = sorted(label_values, key=sort_key)
    label2id = {label: idx for idx, label in enumerate(label_list)}

    print(f"Label mapping:")
    for label, idx in label2id.items():
        print(f"  {label} -> {idx}")

    def _encode(example, *, labels=label2id, column=label_column):
        return {"labels": labels[example[column]]}

    encoded = DatasetDict({split: ds.map(_encode) for split, ds in datasets.items()})
    return encoded, label2id, label_column


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies.

    Args:
        dataset: Training dataset with 'labels' column

    Returns:
        Tensor of class weights (one per class)
    """
    from collections import Counter

    labels = dataset["labels"]
    label_counts = Counter(labels)
    num_classes = len(label_counts)

    total = sum(label_counts.values())

    weights = torch.zeros(num_classes)
    for label, count in label_counts.items():
        weights[label] = total / (num_classes * count)

    print(f"\nClass weights (inverse frequency):")
    for label in sorted(label_counts.keys()):
        print(f"  Class {label}: {weights[label]:.4f} (n={label_counts[label]:,})")

    return weights


class WeightedLossTrainer(Trainer):
    """Custom Trainer that applies class weights to the loss function."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weights = self.class_weights
        if weights is not None:
            weights = weights.to(device=logits.device, dtype=logits.dtype)

        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_dataset(datasets: DatasetDict, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH) -> DatasetDict:
    """Tokenize all text in the dataset."""
    max_len = min(max_length, getattr(tokenizer, "model_max_length", max_length))

    print(f"\nTokenizing dataset (max_length={max_len})...")

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=max_len)

    tokenized = DatasetDict()
    for split, ds in datasets.items():
        print(f"  Tokenizing {split}...")
        tokenized_split = ds.map(_tokenize, batched=True)
        keep = {"input_ids", "attention_mask", "labels"}
        drop_columns = [col for col in tokenized_split.column_names if col not in keep]
        if drop_columns:
            tokenized_split = tokenized_split.remove_columns(drop_columns)
        tokenized[split] = tokenized_split

    return tokenized


# ---------------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------------

def prepare_model(
    num_labels: int,
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
    dtype: torch.dtype,
    base_model_path: Optional[str] = None,
) -> AutoModelForSequenceClassification:
    """Prepare model with LoRA adapters for sequence classification."""
    print(f"\nPreparing model with {num_labels} labels...")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_dtype = dtype if dtype is not None else torch.bfloat16
    model_path = base_model_path or DEFAULT_BASE_MODEL

    print(f"Loading base model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        dtype=resolved_dtype,
        low_cpu_mem_usage=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.id2label = {idx: label for label, idx in label2id.items()}
    model.config.label2id = label2id

    print(f"Adding LoRA adapters (r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT})...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return peft_model


def compute_metrics(eval_prediction) -> Dict[str, float]:
    """Compute accuracy metric for evaluation."""
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    labels = eval_prediction.label_ids
    return {"accuracy": float((predictions == labels).mean())}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    dataset_path: str,
    label_column: str,
    output_dir: str,
    positive_token: str = "SELECT",
    negative_token: str = "PASS",
    null_model: bool = False,
    use_class_weights: bool = False,
    use_stratified_split: bool = False,
    prompt_variable: Optional[List[str]] = None,
    seed: Optional[int] = None,
    base_model_path: Optional[str] = None,
) -> None:
    """
    Train a sequence classification model on the specified dataset.

    Args:
        dataset_path: Path to JSONL dataset file (output of create_y.py)
        label_column: Name of the label column to train on
        output_dir: Directory to save model and outputs
        positive_token: Token for positive class (default "SELECT")
        negative_token: Token for negative class (default "PASS")
        null_model: If True, replace all text with empty strings (baseline model)
        use_class_weights: Apply class weights to loss function for imbalanced data
        use_stratified_split: Use stratified splitting to maintain class ratios
        prompt_variable: List of variable specifications with optional mapping
        seed: Random seed for reproducibility (default 42)
        base_model_path: Path to base model (default: Llama-3.1-8B-Instruct)
    """
    seed_to_use = seed if seed is not None else SEED
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device_str == "cuda" else torch.float32

    print("\n" + "=" * 80)
    print("FINE-TUNING CONFIGURATION")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Label column: {label_column}")
    print(f"Output directory: {output_dir}")
    print(f"Null model: {null_model}")
    print(f"Seed: {seed_to_use}")
    print(f"Base model: {base_model_path or DEFAULT_BASE_MODEL}")
    print("=" * 80)

    # Load tokenizer
    tokenizer_path = base_model_path or DEFAULT_BASE_MODEL
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load and prepare datasets
    datasets = load_and_prepare_datasets(
        dataset_path=dataset_path,
        label_column=label_column,
        positive_token=positive_token,
        negative_token=negative_token,
        prompt_variable=prompt_variable,
        use_stratified_split=use_stratified_split,
        seed=seed_to_use,
    )

    # Encode labels
    datasets, label2id, _ = encode_labels(datasets, label_column="label")

    # Replace text with empty strings for null model
    if null_model:
        print("\nNull model mode: Replacing all text with empty strings...")
        for split in datasets:
            datasets[split] = datasets[split].map(lambda example: {"text": " "})
        print(f"  Replaced text in {len(datasets)} splits")

    # Tokenize
    tokenized = tokenize_dataset(datasets, tokenizer, max_length=MAX_LENGTH)

    # Prepare model
    model = prepare_model(
        num_labels=len(label2id),
        tokenizer=tokenizer,
        label2id=label2id,
        dtype=dtype,
        base_model_path=base_model_path,
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # Compute class weights if requested
    class_weights = None
    if use_class_weights:
        print("\nComputing class weights for weighted loss...")
        class_weights = compute_class_weights(tokenized["train"])
        if torch.cuda.is_available():
            class_weights = class_weights.to(device='cuda', dtype=dtype)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_dir=str(output_path / "logs"),
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=[],
        seed=seed_to_use,
        bf16=(dtype == torch.bfloat16),
        remove_unused_columns=True,
        max_grad_norm=1.0,
    )

    # Initialize trainer
    TrainerClass = WeightedLossTrainer if use_class_weights else Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized.get("validation"),
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    if use_class_weights:
        trainer_kwargs["class_weights"] = class_weights

    trainer = TrainerClass(**trainer_kwargs)

    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    trainer.train()

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATING")
    print("=" * 80)
    metrics = trainer.evaluate() if tokenized.get("validation") is not None else {}

    # Save model
    print("\nSaving model...")
    trainer.save_model()

    # Save metrics
    if metrics:
        metrics_path = output_path / "eval_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
        print(f"Validation accuracy: {metrics.get('eval_accuracy', 'N/A')}")

    # Save tokenizer
    tokenizer.save_pretrained(str(output_path / "tokenizer"))

    # Export datasets with original structure
    print("\nExporting datasets...")
    for split in ["train", "validation", "test"]:
        if split in datasets:
            ds = datasets[split]
            if split in tokenized:
                label_ids = tokenized[split]["labels"]
                ds = ds.add_column("label_id", label_ids)
            output_file = output_path / f"{split}_dataset.jsonl"
            ds.to_json(output_file, orient="records", lines=True)
            print(f"  Saved {split}: {output_file} ({len(ds)} samples)")

    # Cleanup
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaMA model for sequence classification on JSONL dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python finetuning.py \\
    --dataset-path out/y_labels/before_labels.jsonl \\
    --label-column y_title_item_details \\
    --output-dir out/models/before \\
    --positive-token SELECT \\
    --negative-token PASS

  # Train null model (baseline with empty text)
  python finetuning.py \\
    --dataset-path out/y_labels/before_labels.jsonl \\
    --label-column y_title_item_details \\
    --output-dir out/models/before_null \\
    --null-model
        """
    )

    parser.add_argument("--dataset-path", required=True, help="Path to JSONL dataset file")
    parser.add_argument("--label-column", required=True, help="Name of the label column")
    parser.add_argument("--output-dir", required=True, help="Directory to save model and outputs")
    parser.add_argument("--positive-token", default="SELECT", help="Token for positive class (default: SELECT)")
    parser.add_argument("--negative-token", default="PASS", help="Token for negative class (default: PASS)")
    parser.add_argument("--null-model", action="store_true", help="Train null model (empty text baseline)")
    parser.add_argument("--use-class-weights", action="store_true", help="Apply class weights to loss function")
    parser.add_argument("--use-stratified-split", action="store_true", help="Use stratified train/test splitting")
    parser.add_argument("--prompt-variable", type=str, nargs='*', default=None,
                        help="Variable specifications from payload (e.g., 'title=english_name')")
    parser.add_argument('--base-model-path', type=str, default=None,
                        help='Path to base model (default: Llama-3.1-8B-Instruct)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: 42)')

    args = parser.parse_args()

    if not Path(args.dataset_path).exists():
        parser.error(f"Dataset file not found: {args.dataset_path}")

    train(
        dataset_path=args.dataset_path,
        label_column=args.label_column,
        output_dir=args.output_dir,
        positive_token=args.positive_token,
        negative_token=args.negative_token,
        null_model=args.null_model,
        use_class_weights=args.use_class_weights,
        use_stratified_split=args.use_stratified_split,
        prompt_variable=args.prompt_variable,
        base_model_path=args.base_model_path,
        seed=args.seed,
    )
