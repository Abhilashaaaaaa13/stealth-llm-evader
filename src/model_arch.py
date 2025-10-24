import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def setup_model(config: dict):
    """Load base model with LoRA for fine-tuning – memory efficient."""
    model_name = config['model']['base']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Memory efficient load: fp16 + 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Half precision – halves memory
        load_in_8bit=True,  # 8-bit quantization – fits in 4GB VRAM
        device_map="auto",  # Auto GPU placement
    )

    # Add LoRA (smaller rank for memory)
    lora_config = LoraConfig(
        r=config['model']['lora_rank'],
        lora_alpha=config['model']['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # For Mistral
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    # Device move (GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded on {device} with {torch.cuda.get_device_properties(0).total_memory / 1e9 if device == 'cuda' else 'N/A'} GB VRAM")

    return model, tokenizer

def setup_ensemble(config: dict):
    """Setup multiple models for ensemble."""
    models = []
    for name in config['model']['ensemble_models']:
        temp_config = config.copy()
        temp_config['model']['base'] = name
        model, _ = setup_model(temp_config)
        models.append(model)
    return models