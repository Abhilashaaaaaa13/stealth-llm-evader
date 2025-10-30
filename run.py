#!/usr/bin/env python3
import os
import argparse
import logging
import random
import time
from dotenv import load_dotenv
from src.post_process import apply_post_processing
from src.eval import run_evaluation
from src.humanize import inject_human_noise

# -----------------------------------------------------
# Setup
# -----------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------
def save_output(output_path, text):
    """Save text to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def load_config(config_path):
    """Load YAML config or fallback to empty dict."""
    if not os.path.exists(config_path):
        return {}
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------
# Main pipeline
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="🧠 Essay Generator — API-only, humanized output")
    parser.add_argument("--mode", type=str, required=True, choices=["generate", "evaluate"])
    parser.add_argument("--prompt", type=str, default=None, help="Topic (short phrase)")
    parser.add_argument("--output", type=str, default="outputs/output.txt")
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    parser.add_argument("--evasion_config", type=str, default="config/evasion_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    evasion_config = load_config(args.evasion_config)
    model_config = config.get("model", {})

    logger.info(f"🚀 Running in {args.mode.upper()} mode...")

    # -----------------------------------------------------
    # MODE: GENERATE
    # -----------------------------------------------------
    if args.mode == "generate":
        if not args.prompt:
            logger.error("Provide a short topic with --prompt, e.g. --prompt=\"impact of AI on education\"")
            return

        try:
            # Use Groq API client (API-only)
            from groq import Groq
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("❌ GROQ_API_KEY missing. Add it to your .env file.")

            client = Groq(api_key=groq_api_key)
            groq_model = model_config.get("groq_model", "llama-3.1-8b-instant")

            # ---------------------------
            # Rich, contest-focused system prompt
            # ---------------------------
            generation_prompt_template = """
You are an experienced human essayist and creative researcher.
Write a long-form essay on the topic: "{topic}".

Style & voice:
- Intelligent, humane, reflective — similar to a feature article (not a speech).
- Keep paragraphs as the primary structural unit (plain paragraphs separated by one blank line).
- Vary sentence length and rhythm. Use occasional idioms, metaphors, subtle rhetorical questions.
- Include 2–4 short factual references (study title or organization + year or short stat); weave them naturally as inline citations (e.g., "a 2019 UNESCO report", "a 2017 study by Smith et al."). Do NOT fabricate elaborate studies — prefer well known named organizations or general references if unsure.
- Avoid repetitive "In conclusion" style phrases. End with a subtle, reflective final paragraph (no explicit "conclusion" header).
- Keep first-person to a minimum (a restrained "one might say" or "I have noticed" once or twice at most).
- Insert 2–3 short 1–2 sentence paragraphs for emphasis somewhere in the essay.
- Paragraph length: typically 4–8 sentences; occasionally 1–2 sentence paragraphs for emphasis.
- Do NOT output lists, numbered headings, or markdown; write continuous paragraphs.

Requirements:
- Word length: aim ≈ 2500 words (acceptable 2000–3000). Continue until the essay is fully developed.
- Relevance: remain tightly focused on the topic.
- Creativity: include at least two short "out-of-the-box" syntheses (one-sentence speculative ideas).
- Clean output: no underscores, no stray markup, correct capitalization for "I", correct spacing and punctuation.

Write the essay directly in plain paragraphs. Do not produce any meta commentary (e.g., "As an AI...").
"""

            continuation_prompt = """
Continue the essay naturally from where you left off. Maintain the same reflective voice and paragraph rules.
Add new ideas, fresh examples, or a short creative insight (an unconventional implication).
Do not repeat earlier wording; build upon earlier points.
Stop only when the essay feels complete (~2500 words total).
"""

            def generate_chunk(prompt_text, context_text=None):
                """Request a creative text chunk from Groq. If context_text is provided, include it so the model continues coherently."""
                # construct messages: if context_text present, pass it as assistant content so model continues
                messages = []
                if context_text:
                    messages.append({"role": "assistant", "content": context_text})
                messages.append({"role": "user", "content": prompt_text})
                resp = client.chat.completions.create(
                    model=groq_model,
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.9,
                    top_p=0.93,
                    stream=False,
                )
                return resp.choices[0].message.content.strip()

            # --- first chunk
            logger.info("✍️ Generating initial essay chunk...")
            essay = generate_chunk(generation_prompt_template.format(topic=args.prompt))
            word_count = len(essay.split())
            logger.info(f"Generated {word_count} words so far.")

            # --- continue until target (safe loop with an upper bound)
            attempts = 0
            while word_count < 2400 and attempts < 6:
                attempts += 1
                logger.info(f"✍️ Extending essay ({word_count} words so far)...")
                # pass last part of essay as context for continuity
                context_tail = essay[-1600:] if len(essay) > 1600 else essay
                addition = generate_chunk(continuation_prompt, context_text=context_tail)
                # small cleanup of chunk
                addition = addition.replace("_", "").replace("*", "")
                essay += "\n\n" + addition
                word_count = len(essay.split())
                logger.info(f"... now {word_count} words.")
                time.sleep(1.2)  # polite pacing

            # final cleanup
            essay = essay.replace("_", "").replace("*", "")
            # ensure " I " capitalization
            essay = essay.replace(" i ", " I ")
            # remove stray duplicated spaces & normalize line breaks
            essay = "\n\n".join([p.strip() for p in essay.splitlines() if p.strip() != ""])
            logger.info(f"✅ Final essay length: {word_count} words")

        except Exception as e:
            logger.error(f"⚠️ Generation failed: {e}")
            return

        # -----------------------------------------------------
        # Post-processing for human realism + contest checklist
        # -----------------------------------------------------
        if evasion_config.get("evasion", {}).get("human_noise", True):
            essay = inject_human_noise(essay, evasion_config)
            logger.info("✨ Injected subtle human noise.")

        essay = apply_post_processing(essay, evasion_config)

        # Final safety tweaks
        essay = essay.replace("_", "").replace("*", "")
        # Ensure single blank line between paragraphs
        essay = "\n\n".join([p.strip() for p in essay.split("\n\n") if p.strip()])

        save_output(args.output, essay)
        logger.info(f"✅ Essay saved ({word_count} words) ➜ {args.output}")

    # -----------------------------------------------------
    # MODE: EVALUATE
    # -----------------------------------------------------
    elif args.mode == "evaluate":
        run_evaluation(config, evasion_config)


# -----------------------------------------------------
# Entry point
# -----------------------------------------------------
if __name__ == "__main__":
    main()
