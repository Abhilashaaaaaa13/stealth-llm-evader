from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def setup_model(config: dict):
    """Load base model with LoRA for fine-tuning."""
    model_name = config['model']['base']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add LoRA
    lora_config = LoraConfig(
        r=config['model']['lora_rank'],
        lora_alpha=config['model']['lora_alpha'],  # Comma added here
        target_modules=["c_attn"],  # For GPT2; adjust for Llama
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer

def setup_ensemble(config: dict):
    """Setup multiple models for ensemble."""
    models = []
    for name in config['model']['ensemble_models']:
        # Temp override for testing
        temp_config = config.copy()
        temp_config['model']['base'] = name
        model, _ = setup_model(temp_config)
        models.append(model)
    return models