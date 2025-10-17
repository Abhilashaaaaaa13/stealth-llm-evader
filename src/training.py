from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

def fine_tune(model, tokenizer , dataset: Dataset, epochs: int, config: dict):
    """Fine-tune with SFT (Supervised Fine-tuning)."""
    training_args = TrainingArguments(
        output_dir='models/fine_tuned',
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
    )
    trainer.train()
    return model