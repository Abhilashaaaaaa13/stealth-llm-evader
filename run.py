"""Main entry point for Stealth LLM Evader.
Modes : train, generate,eval."""
import argparse
import yaml
import logging
from pathlib import Path
import torch

from src.data_prep import load_dataset
from src.model_arch import setup_model, setup_ensemble
from src.training import fine_tune
from src.generation import generate_text
from src.post_process import apply_post_processing
from src.ensemble import ensemble_large
from src.eval import run_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path,'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Stealth LLM Pipeline")
    parser.add_argument('--mode', choices=['train','generate','eval'],required=True, help = "Run mode")
    parser.add_argument('--config', default='config/model_config.yaml',help="Model config")
    parser.add_argument('--evasion-config', default='config/evasion_config.yaml',help="Evasion config")
    parser.add_argument('--prompt',  help = "Prompt for generation")
    parser.add_argument('--output', default='output/generated.txt', help = "Output path")
    parser.add_argument('--input', default='output/generated_samples', help = "Input dir for eval")
    parser.add_argument('--epochs', type=int, default=1, help = "Training epochs")
    args = parser.parse_args()

    model_config = load_config(args.config)
    evasion_config = load_config(args.evasion_config)

    if args.mode == 'train':
        dataset = load_dataset('data/human_texts.json')
        model, tokenizer = setup_model(model_config)
        fine_tuned_model = fine_tune(model, tokenizer , dataset, epochs = args.epochs  , config=model_config)
        fine_tuned_model.save_pretrained('models/fine_tuned')
        logger.info("Training complete! Checkpoint saved.")

    elif args.mode == 'generate':
        if not args.prompt:
            logger.error("Prompt required for generate mode")
            return
        models = setup_ensemble(model_config)
        drafts = [generate_text(model, args.prompt, model_config['evasion']['max_length']) for model in models]
        ensembled = ensemble_merge(drafts)
        final_text = apply_post_processing(ensembled, evasion_config)
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(final_text)
        logger.info(f"Generated stealthy text: {output_path} (~{len(final_text.split())} words)")

if __name__ == '__main__':
    main()
