import torch
from transformers import AutoTokenizer

def generate_text(model, prompt: str, max_length: int, tokenizer=None):
    """Generate text from prompt."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):]