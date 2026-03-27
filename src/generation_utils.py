from typing import Optional, Dict, List, Any
import ast
import torch
from transformers import LogitsProcessor


def normalize_item_details(value: Any) -> str:
    """Normalize item_details field to a single string value.
    
    Handles various input formats:
    - String representation of list (e.g., "['a', 'b']") -> first element
    - Actual list/tuple -> first element
    - Regular string -> as-is
    - None -> empty string
    
    Args:
        value: The item_details value to normalize.
        
    Returns:
        Normalized string value.
    """
    if value is None:
        return ""
    
    # If it's a string that looks like a list, try to parse it
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('[') or stripped.startswith('('):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return str(parsed[0]) if parsed else ""
            except (ValueError, SyntaxError):
                pass  # Keep as string if parsing fails
        return value
    
    # If it's already a list or tuple, take first element
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    
    return str(value)


def preprocess_variables_list(variables_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Preprocess a list of variable dictionaries for prompt creation.
    
    Handles special cases like item_details being a list/tuple or string
    representation of a list.
    
    Args:
        variables_list: List of variable dictionaries.
        
    Returns:
        Processed list with normalized values.
    """
    if not variables_list:
        return []
    
    # Check if any record has item_details
    has_item_details = any("item_details" in variables for variables in variables_list)
    
    if not has_item_details:
        return variables_list
    
    processed = []
    for variables in variables_list:
        if "item_details" in variables:
            item_details = variables.get("item_details")
            normalized = normalize_item_details(item_details)
            processed.append({**variables, "item_details": normalized})
        else:
            processed.append(variables)
    
    return processed


def prompt_creator(
    variables: dict,
    positive_token: str = "SELECT",
    negative_token: str = "PASS",
    prompt_template: Optional[str] = None,
) -> tuple[str, str]:
    """
    Create a prompt for the purchase decision oracle.

    Args:
        variables: Dictionary of product attributes (e.g., {"title": "...", "description": "..."})
        positive_token: Token representing positive decision (default "SELECT")
        negative_token: Token representing negative decision (default "PASS")
        prompt_template: Optional custom prompt template string. If provided, must contain
                        {product_info}, {positive_token}, and {negative_token} placeholders.
                        If None, uses default Etsy browsing assistant prompt.

    Returns:
        A tuple containing:
        - The full prompt string to be fed to the model.
        - The product information string.
    """
    product_info = "\n".join(f"{key}: {value}" for key, value in variables.items())

    if prompt_template:
        # Use custom template, replacing placeholders
        full_prompt = prompt_template.format(
            product_info=product_info,
            positive_token=positive_token,
            negative_token=negative_token
        )
    else:
        # Use default prompt (V4 - Etsy browsing assistant)
        preamble = f"""System: You are helping someone browse Etsy. Only {positive_token} items they would genuinely appreciate and want to see - be very selective.

    {positive_token} only if the item is EXCEPTIONAL and would make someone say "wow, that's special":
    - Truly beautiful, impressive, or emotionally resonant
    - Exceptional craftsmanship, artistic merit, or historical significance
    - Something you'd be excited to own, gift, or show others
    - Stands out as memorable among thousands of items

    {negative_token} for everything else, including:
    - Ordinary vintage items without special appeal
    - Generic handmade items lacking wow factor
    - Mass-produced or common items
    - Anything that's just "okay" or "fine" but not exciting
    - Items where you'd scroll past without a second thought

    Be highly selective - most items should be {negative_token}. Only {positive_token} items that truly deserve attention and would be genuinely appreciated. Output only: {positive_token} or {negative_token}.

    Product Details:
    """
        conclusion = "Decision:"
        full_prompt = f"{preamble}\n{product_info}\n{conclusion}"

    return full_prompt, product_info


class SanitizeLogitsProcessor(LogitsProcessor):
    def __init__(self, min_logit: float = -50.0):
        self.min_logit = min_logit

    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=self.min_logit, neginf=self.min_logit, posinf=0.0)
        return torch.clamp(scores, min=self.min_logit)
    

class PosNegLogitsProcessor(LogitsProcessor):
    def __init__(self, positive, negative, tokenizer, initial_length=None, debug=False):
        self.positive = positive
        self.negative = negative
        self.tokenizer = tokenizer
        self.initial_length = initial_length  # Can be None - will be set on first call
        self.debug = debug
        self.first_call = True
        
        self.tokenized_pos = self.tokenizer.encode(self.positive, add_special_tokens=False)
        self.tokenized_neg = self.tokenizer.encode(self.negative, add_special_tokens=False)
        if len(self.tokenized_pos) != 1 or len(self.tokenized_neg) != 1:
            raise ValueError("Positive and Negative tokens must each tokenize to a single token ID.")
        
        # Store probabilities for retrieval after generation
        self.last_probabilities = []  # List of (pos_prob, neg_prob) tuples for each sample in batch
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids shape: (batch_size, sequence_length)
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        
        # On first call, set initial_length if not provided
        if self.first_call:
            if self.initial_length is None:
                self.initial_length = current_length
            self.first_call = False
            if self.debug:
                print(f"First call - setting initial_length to {self.initial_length}")
        
        # Calculate how many tokens have been generated beyond the prompt
        tokens_generated = current_length - self.initial_length
        
        if self.debug:
            print(f"Batch size: {batch_size}, Current length: {current_length}, Initial: {self.initial_length}, Generated: {tokens_generated}")
            if tokens_generated > 0:
                print(f"Last generated tokens: {input_ids[:, -min(tokens_generated, 5):]}")
        
        # If we already generated one or more tokens, force EOS
        if tokens_generated >= 1:
            if self.debug:
                print(f"Forcing EOS after {tokens_generated} token(s)")
            scores.fill_(-float('inf'))
            scores[:, self.tokenizer.eos_token_id] = 1e4
            return scores
            
        # First generation step: only allow YES or NO (not EOS yet)
        pos_tokens = self.tokenized_pos
        neg_tokens = self.tokenized_neg
        
        if self.debug:
            print(f"First generation - allowing only YES ({pos_tokens[0]}) or NO ({neg_tokens[0]})")
            print(f"EOS token ID: {self.tokenizer.eos_token_id}")
        
        # Extract original logits for yes/no
        pos_neg_logits = scores[:, [pos_tokens[0], neg_tokens[0]]]
        
        if self.debug:
            print(f"Pos/Neg logits: {pos_neg_logits}")
        
        # Get probabilities using softmax
        pos_neg_probs = torch.nn.functional.softmax(pos_neg_logits, dim=-1)
        
        pos_prob = pos_neg_probs[:, 0]
        neg_prob = pos_neg_probs[:, 1]
        
        # Store probabilities for later retrieval (convert to CPU and list)
        self.last_probabilities = list(zip(pos_prob.cpu().tolist(), neg_prob.cpu().tolist()))
        
        if self.debug:
            print(f"Positive prob: {pos_prob}")
            print(f"Negative prob: {neg_prob}")
        
        # Mask all tokens with -inf
        scores.fill_(-float('inf'))
        
        # Set the higher probability choice to a large value
        # Do NOT allow EOS on this step - only YES or NO
        yes_mask = pos_prob > neg_prob
        high_val = torch.tensor(1e4, device=scores.device, dtype=scores.dtype)
        low_val = torch.tensor(-float('inf'), device=scores.device, dtype=scores.dtype)
        scores[:, pos_tokens[0]] = torch.where(yes_mask, high_val, low_val)
        scores[:, neg_tokens[0]] = torch.where(~yes_mask, high_val, low_val)
        
        return scores
