import torch
from transformers import AutoTokenizer

def generate_text(model, prompt: str, max_length: int, tokenizer=None):
    """
    Generate text from prompt – fixed for longer outputs.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    
    # Device setup for GPU (if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Generating on {device}")
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU
    input_len = inputs['input_ids'].shape[1]  # Prompt tokens count
    
    # Limit new tokens to model max - input (avoid overflow)
    model_max = model.config.max_position_embeddings or 4096  # Default for Mistral
    max_new = min(max_length, model_max - input_len)
    print(f"Generating up to {max_new} new tokens on {device}...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,  # New tokens only
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
            early_stopping=False,
            num_beams=1,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()  # Strip prompt + clean whitespace