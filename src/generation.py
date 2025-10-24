import torch
from transformers import AutoTokenizer

def generate_text(model, prompt: str, max_length: int, tokenizer=None):
    """Generate text from prompt-fixed for longer outputs"""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
            early_stopping=False,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):]