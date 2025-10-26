import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def setup_model(config: dict):
    """Load base model with LoRA for fine-tuning – memory efficient for 4GB VRAM."""
    model_name = config['model']['base']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # BitsAndBytesConfig for 4-bit quantization (better for low VRAM)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit for ~3GB VRAM usage
        bnb_4bit_compute_dtype=torch.float16,  # FP16 compute for speed
        bnb_4bit_use_double_quant=True,  # Extra memory saving
        llm_int8_enable_fp32_cpu_offload=True,  # CPU fallback if VRAM full
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,  # Quantization config
        dtype=torch.float16,  # Half precision
        device_map="auto",  # Auto GPU/CPU split
        trust_remote_code=True,  # For Mistral if needed
    )

    # Add LoRA (small rank for memory)
    lora_config = LoraConfig(
        r=config['model']['lora_rank'],
        lora_alpha=config['model']['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # For Mistral
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    # Print device info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vram_info = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if device == 'cuda' else 'N/A'
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