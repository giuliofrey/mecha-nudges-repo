"""
Step 1: Label Generation via Constrained LLM Inference.

Prompts a causal language model (default: Gemma-3-27B) with product information
and forces a binary decision (e.g., SELECT/PASS) using constrained generation.
The PosNegLogitsProcessor masks all vocabulary tokens except the two decision
tokens, then forces EOS on the next step to guarantee exactly one output token.

Output: JSONL with binary labels, raw text, and decision probabilities per listing.

Usage:
    python create_y.py \\
        --input data/etsy_before.jsonl \\
        --output out/y_labels/before_labels.jsonl \\
        --input-length 100000 \\
        --variable Title=title Item_Details=item_details \\
        --positive-token SELECT --negative-token PASS \\
        --batch-size 8 \\
        --prompt-template-file prompts/prompt_v4.txt
"""

import argparse
from typing import Optional, Tuple, List, Dict, Any
import sys
import re
from pathlib import Path

from generation_utils import prompt_creator, preprocess_variables_list, PosNegLogitsProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LogitsProcessorList
import torch
from tqdm.auto import tqdm
import json


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-3-27b-it"
DEFAULT_POSITIVE_TOKEN = "SELECT"
DEFAULT_NEGATIVE_TOKEN = "PASS"
MAX_ITEM_DETAILS_LENGTH = 5000  # Maximum characters for item_details field

dtype = torch.bfloat16
torch.backends.cuda.matmul.allow_tf32 = True

# Only include valid generation parameters for greedy decoding
GENERATION_KWARGS = {
    "max_new_tokens": 15,
    "do_sample": False,
    "return_full_text": False,
}


# ---------------------------------------------------------------------------
# Helpers (inlined from nudges.tools and nudges.inference)
# ---------------------------------------------------------------------------

def parse_decision(text: str, positive="SELECT", negative="PASS") -> Optional[bool]:
    """Parse model output text into a boolean decision."""
    negative_no_underscore = negative.replace("_", " ")
    t = text.strip().upper()
    if t == positive:
        return True
    elif t == negative or t == negative_no_underscore:
        return False
    m = re.search(rf"\b(?:{positive}|{negative}|{negative_no_underscore})\b", t)
    if m is None:
        return None
    elif m.group(0) == positive:
        return True
    elif m.group(0) == negative or m.group(0) == negative_no_underscore:
        return False
    return None


def sanitize_label(label: str) -> str:
    """Sanitize a label string to create valid column names."""
    sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
    parts = [part for part in sanitized.split("_") if part]
    return "_".join(parts) or "field"


def label_column_names(variable_mappings: List[Tuple[str, str]]) -> Tuple[str, str]:
    """Generate column names for label and raw output based on variable mappings."""
    suffix = "_".join(sanitize_label(label) for label, _ in variable_mappings)
    suffix = suffix or "decision"
    base = f"y_{suffix}"
    return base, f"{base}_raw"


def parse_variable_args(variable_args: List[str]) -> List[Tuple[str, str]]:
    """Parse variable argument strings into (label, column) tuples."""
    mappings: List[Tuple[str, str]] = []
    for raw in variable_args:
        if "=" in raw:
            label, column = raw.split("=", 1)
        else:
            label, column = raw, raw
        label = label.strip()
        column = column.strip()
        if not label or not column:
            raise ValueError(f"Invalid variable specification '{raw}'. Use 'Label=column'.")
        mappings.append((label, column))
    if not mappings:
        raise ValueError("At least one variable must be specified.")
    return mappings


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pipeline(model_name: str, gpu_id: int = 0):
    """Load a text-generation pipeline on a single GPU."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(f"cuda:{gpu_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=gpu_id,
    )

    if getattr(gen_pipeline.model.config, "pad_token_id", None) is None:
        gen_pipeline.model.config.pad_token_id = tokenizer.eos_token_id

    return gen_pipeline


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_batch_decisions(
    gen_pipeline,
    variables_list: List[Dict[str, str]],
    positive_token: str = DEFAULT_POSITIVE_TOKEN,
    negative_token: str = DEFAULT_NEGATIVE_TOKEN,
    batch_size: int = 1,
    prompt_template: Optional[str] = None,
) -> Tuple[List[Optional[bool]], List[str], List[str], List[Tuple[float, float]]]:
    """Generate batch decisions for a list of variable dictionaries.

    Args:
        gen_pipeline: HuggingFace text-generation pipeline
        variables_list: List of dictionaries containing product variables
        positive_token: Token for positive decision
        negative_token: Token for negative decision
        batch_size: Batch size for generation
        prompt_template: Optional custom prompt template with placeholders

    Returns:
        Tuple of (decisions, raw_texts, prompts, probabilities)
        where probabilities is a list of (pos_prob, neg_prob) tuples
    """
    if not variables_list:
        return [], [], [], []

    # Preprocess variables (handles item_details normalization)
    variables_list = preprocess_variables_list(variables_list)

    # Create prompts from product variables
    _prompt_tuples = [
        prompt_creator(variables, positive_token=positive_token, negative_token=negative_token, prompt_template=prompt_template)
        for variables in variables_list
    ]
    prompts = [p[0] for p in _prompt_tuples]

    # Constrained generation: force output to be exactly one of the two decision tokens
    logits_processor = PosNegLogitsProcessor(
        positive=positive_token,
        negative=negative_token,
        tokenizer=gen_pipeline.tokenizer,
    )

    outputs = gen_pipeline(
        prompts,
        batch_size=min(len(prompts), batch_size),
        **GENERATION_KWARGS,
        logits_processor=LogitsProcessorList([logits_processor]),
    )

    decisions: List[Optional[bool]] = []
    raw_texts: List[str] = []
    for output in outputs:
        text = output[0]["generated_text"].strip()
        decision = parse_decision(text, positive=positive_token, negative=negative_token)
        if decision is None:
            raise ValueError(
                f"Invalid model response: expected '{positive_token}' or '{negative_token}', "
                f"but got '{text}'. This indicates the logits processor failed to constrain generation properly."
            )
        decisions.append(decision)
        raw_texts.append(text)

    # Retrieve probabilities from the logits processor
    probabilities = logits_processor.last_probabilities if logits_processor.last_probabilities else [(0.5, 0.5)] * len(decisions)

    return decisions, raw_texts, prompts, probabilities


# ---------------------------------------------------------------------------
# Main labeling loop
# ---------------------------------------------------------------------------

def create_y(
    gen_pipeline,
    input_path: str,
    input_length: Optional[int],
    output_path: str,
    *,
    variable_mappings: List[Tuple[str, str]],
    positive_token: str = DEFAULT_POSITIVE_TOKEN,
    negative_token: str = DEFAULT_NEGATIVE_TOKEN,
    batch_size: int = 1,
    prompt_template: Optional[str] = None,
) -> None:
    """Generate labels for input data using the model.

    Args:
        gen_pipeline: HuggingFace text-generation pipeline
        input_path: Path to input JSONL file
        input_length: Total number of records in input (for progress tracking)
        output_path: Path to output JSONL file
        variable_mappings: List of (label, column) tuples for prompt variables
        positive_token: Token for positive decision
        negative_token: Token for negative decision
        batch_size: Number of samples per generation call
        prompt_template: Optional custom prompt template with placeholders
    """
    if not variable_mappings:
        raise ValueError("At least one variable mapping must be provided.")

    label_column, raw_column = label_column_names(variable_mappings)
    input_file = Path(input_path)
    output_file = Path(output_path)

    # Progress bar
    progress_total = input_length if input_length and input_length > 0 else None
    progress = tqdm(total=progress_total, desc="Generating labels", unit="sample",
                    postfix={"pos": 0, "neg": 0, "pos%": 0.0})

    payload_originals: List[Dict[str, str]] = []
    records: List[Dict[str, Any]] = []
    positive_count = 0
    negative_count = 0

    with input_file.open("r", encoding="utf-8") as in_file, \
         output_file.open("w", encoding="utf-8") as out_file:

        for line_num, line in enumerate(in_file, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"\nWarning: Skipping malformed JSON at line {line_num}: {e}", file=sys.stderr)
                continue

            record_id = record.get("id")

            # Extract payload (label-based, for prompting) and payload_original (original column names)
            payload = {label: record.get(column, "") for label, column in variable_mappings}
            payload_original = {column: record.get(column, "") for _, column in variable_mappings}

            # Truncate item_details if it exceeds maximum length
            if 'item_details' in payload:
                item_details_value = payload['item_details']
                if isinstance(item_details_value, (list, tuple)):
                    actual_string = str(item_details_value[0]) if item_details_value else ""
                else:
                    actual_string = str(item_details_value) if item_details_value else ""

                if len(actual_string) > MAX_ITEM_DETAILS_LENGTH:
                    truncated = actual_string[:MAX_ITEM_DETAILS_LENGTH]
                    if isinstance(item_details_value, (list, tuple)):
                        payload['item_details'] = [truncated]
                    else:
                        payload['item_details'] = truncated

            records.append({"id": record_id, "payload": payload, "payload_original": payload_original})
            payload_originals.append(payload)

            # Process batch when full
            if len(payload_originals) == batch_size:
                batch_decisions, batch_raw, _, batch_probs = generate_batch_decisions(
                    gen_pipeline, payload_originals,
                    positive_token=positive_token, negative_token=negative_token,
                    batch_size=batch_size, prompt_template=prompt_template,
                )

                # Write results
                for meta, decision, raw_text, (pos_prob, neg_prob) in zip(records, batch_decisions, batch_raw, batch_probs):
                    output_record = {
                        "id": meta["id"],
                        label_column: decision,
                        raw_column: raw_text,
                        "prob_true": pos_prob,
                        "prob_false": neg_prob,
                        "payload": meta["payload"],
                        "payload_original": meta["payload_original"],
                    }

                    out_file.write(json.dumps(output_record) + "\n")

                    if decision is True:
                        positive_count += 1
                    elif decision is False:
                        negative_count += 1

                progress.update(len(batch_decisions))
                total_labels = positive_count + negative_count
                pos_pct = (positive_count / total_labels * 100) if total_labels > 0 else 0.0
                progress.set_postfix({"pos": positive_count, "neg": negative_count, "pos%": f"{pos_pct:.1f}"})
                payload_originals.clear()
                records.clear()

        # Process remaining batch
        if payload_originals:
            batch_decisions, batch_raw, _, batch_probs = generate_batch_decisions(
                gen_pipeline, payload_originals,
                positive_token=positive_token, negative_token=negative_token,
                batch_size=len(payload_originals), prompt_template=prompt_template,
            )

            for meta, decision, raw_text, (pos_prob, neg_prob) in zip(records, batch_decisions, batch_raw, batch_probs):
                output_record = {
                    "id": meta["id"],
                    label_column: decision,
                    raw_column: raw_text,
                    "prob_true": pos_prob,
                    "prob_false": neg_prob,
                    "payload": meta["payload"],
                    "payload_original": meta["payload_original"],
                }
                out_file.write(json.dumps(output_record) + "\n")

                if decision is True:
                    positive_count += 1
                elif decision is False:
                    negative_count += 1

            progress.update(len(batch_decisions))

    progress.close()

    # Print label balance statistics
    total_labels = positive_count + negative_count
    if total_labels > 0:
        pos_pct = positive_count / total_labels * 100
        neg_pct = negative_count / total_labels * 100
        print(f"\nLabel Distribution:")
        print(f"  {positive_token}: {positive_count:,} ({pos_pct:.2f}%)")
        print(f"  {negative_token}: {negative_count:,} ({neg_pct:.2f}%)")
        print(f"  Total: {total_labels:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model-based labels for product data.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to output JSONL file with labels.")
    parser.add_argument("--input-length", type=int, required=True, help="Number of rows in the input JSONL file.")
    parser.add_argument(
        "--variable", required=True, nargs="+",
        help="Mapping(s) for prompt variables (e.g. Title=title Item_Details=item_details).",
    )
    parser.add_argument("--positive-token", required=True, help="Token treated as positive decision.")
    parser.add_argument("--negative-token", required=True, help="Token treated as negative decision.")
    parser.add_argument("--batch-size", type=int, default=1, help="Samples per generation call.")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name or path (default: Gemma-3-27B).")
    parser.add_argument(
        "--prompt-template-file", type=str, default=None,
        help="Path to custom prompt template file. Use {product_info}, {positive_token}, {negative_token} as placeholders.",
    )
    args = parser.parse_args()

    try:
        variable_mappings = parse_variable_args(args.variable)
    except ValueError as exc:
        parser.error(str(exc))

    # Load prompt template if provided
    prompt_template = None
    if args.prompt_template_file:
        with open(args.prompt_template_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        print(f"Using custom prompt template from: {args.prompt_template_file}")

    # Load model
    print(f"Loading model: {args.model}")
    gen_pipeline = load_pipeline(args.model)

    create_y(
        gen_pipeline,
        input_path=args.input,
        input_length=args.input_length,
        output_path=args.output,
        variable_mappings=variable_mappings,
        positive_token=args.positive_token,
        negative_token=args.negative_token,
        batch_size=args.batch_size,
        prompt_template=prompt_template,
    )


if __name__ == "__main__":
    main()
