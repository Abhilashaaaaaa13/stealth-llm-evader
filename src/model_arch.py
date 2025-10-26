import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def setup_model(config: dict):
    """Load base model with LoRA for fine-tuning – memory efficient."""
    model_name = config['model']['base']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Memory efficient load: fp16 + 8-bit quantization with CPU offload
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True  # Enables CPU fallback for low VRAM
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,  # New: Use this instead of load_in_8bit
        dtype=torch.float16,  # Changed: Use 'dtype' (not torch_dtype)
        device_map="auto",  # Keeps auto placement (GPU/CPU split)
    )

    # Add LoRA (smaller rank for memory)
    lora_config = LoraConfig(
        r=config['model']['lora_rank'],
        lora_alpha=config['model']['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # For Mistral
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    # No manual .to(device) – device_map="auto" handles it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vram_info = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else 'N/A'
    print(f"Model loaded on {device} with {vram_info} VRAM available")

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