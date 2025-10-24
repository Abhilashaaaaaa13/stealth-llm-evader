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
        lora_alpha=config['model']['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # For Mistral
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    # GPU Move (New – Fix Stuck Load)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded on {device}")

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