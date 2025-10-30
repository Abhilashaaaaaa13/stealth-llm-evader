import streamlit as st
import os
import random
import re
import requests
import json
import time
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
import numpy as np
# We need to explicitly load the .env file to ensure variables are available
from dotenv import load_dotenv

# Load environment variables (like GROQ_API_KEY) right at the start
load_dotenv()

# --- Configuration (Mimicking config files) ---
GROQ_MODEL = "llama-3.1-8b-instant"
TARGET_WORD_COUNT = 2500
MAX_WORD_COUNT = 2700
EVASION_CONFIG = {"evasion": {"human_noise": True}}
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Setup NLTK (Ensure required resources are downloaded) ---
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# --- Sentence Transformer and Synonyms (for Post-Processing) ---
def safe_synonym(word):
    """Return simple synonym only if it's common, short, and meaningful."""
    try:
        syns = [w for w in wordnet.synsets(word) for w in w.lemma_names()]
        syns = [s for s in syns if s.isalpha() and len(s) < 10 and s.lower() != word.lower()]
        if syns:
            return random.choice(syns).replace('_', ' ')
    except:
        pass
    return word

def get_coherence_score(text1, text2):
    """Mock coherence check. In a real environment, this would use SentenceTransformer."""
    # We maintain a high score to ensure post-processing applies
    return 0.95


# --- POST-PROCESSING FUNCTIONS (from post_process.py) ---

def inject_contractions(sentences):
    """Apply natural contractions to make tone conversational."""
    contraction_map = {
        "is not": "isn't", "are not": "aren't", "cannot": "can't",
        "do not": "don't", "did not": "didn't", "it is": "it's",
        "we are": "we're", "you are": "you're", "i am": "I'm",
        "will not": "won’t", "would not": "wouldn’t", "should not": "shouldn’t",
        "has not": "hasn’t", "have not": "haven’t", "could not": "couldn’t"
    }
    updated = []
    for s in sentences:
        for orig, cont in contraction_map.items():
            if random.random() < 0.4:
                s = re.sub(rf"\b{orig}\b", cont, s, flags=re.IGNORECASE)
        updated.append(s)
    return updated

def vary_sentence_lengths(sentences):
    """Add natural variation in sentence rhythm and length."""
    new_sentences = []
    for s in sentences:
        words = s.split()
        if len(words) > 25 and random.random() < 0.45:
            split = len(words) // 2 + random.randint(-4, 4)
            s1 = ' '.join(words[:split]) + random.choice(["...", " —", "."])
            s2 = ' '.join(words[split:]).capitalize()
            new_sentences += [s1, s2]
        elif len(words) < 10 and random.random() < 0.4:
            s += random.choice([
                ", you know.", ", honestly.", ", I guess.", ", in a way.",
                ". Sorta makes sense though.", ", maybe."
            ])
            new_sentences.append(s)
        else:
            new_sentences.append(s)
    return new_sentences

def paraphrase_sentences(sentences):
    """Light synonym replacements for originality, without breaking tone."""
    result = []
    for s in sentences:
        words = s.split()
        for i in range(len(words)):
            if random.random() < 0.12:
                w = re.sub(r'[^\w]', '', words[i].lower())
                new_w = safe_synonym(w)
                if new_w and new_w != w:
                    words[i] = new_w
        result.append(' '.join(words))
    return result

def inject_idioms(sentences):
    """Add idioms in contextually natural ways."""
    idioms = [
        "at the end of the day", "the ball is in your court",
        "every cloud has a silver lining", "it’s not rocket science",
        "hit the nail on the head", "a blessing in disguise",
        "the tip of the iceberg", "goes hand in hand with", "a double-edged sword" # ADDED/UPDATED
    ]
    for i in range(len(sentences)):
        if random.random() < 0.15 and len(sentences[i].split()) > 12:
            idiom = random.choice(idioms)
            if random.random() < 0.5:
                sentences[i] += f", kinda like they say, {idiom}."
            else:
                sentences[i] = f"As they say, {idiom}, " + sentences[i][0].lower() + sentences[i][1:]
    return sentences

def inject_emotional_tone(text):
    """Adds contextual emotion based on keywords."""
    lower_text = text.lower()
    if "climate" in lower_text or "environment" in lower_text:
        endings = [
            "\n\nIt’s hard not to feel a quiet urgency in that truth.",
            "\n\nThere’s something deeply human about wanting to fix what we’ve broken.",
            "\n\nHope, fragile as it seems, still lingers in the wind turbines and solar fields."
        ]
    elif "education" in lower_text or "learning" in lower_text:
        endings = [
            "\n\nMaybe learning has never been about perfection — but evolution.",
            "\n\nSomehow, AI and curiosity are shaping a new kind of classroom: one without walls.",
            "\n\nEducation, in essence, keeps rewriting its own story."
        ]
    elif "psychology" in lower_text or "decision" in lower_text:
        endings = [
            "\n\nOur choices, perhaps, reveal more about our fears than our logic.",
            "\n\nIn business and in life, the mind’s maze never stops surprising us.",
            "\n\nSometimes, it’s not about making the right decision — but understanding why we made it."
        ]
    else:
        endings = [
            "\n\nFunny how thoughts just tumble like that sometimes.",
            "\n\nHonestly, I’m not sure that says everything — but it says enough.",
            "\n\nAnd maybe, that’s just the way people think — imperfectly, yet meaningfully."
        ]
    return text + random.choice(endings)

def add_human_touches(text):
    """Adds light imperfections and tone drifts to mimic human flow."""
    # Lightly adjust I/i capitalization
    text = re.sub(r'\b[Ii]\b', lambda m: "I" if random.random() > 0.25 else "i", text)
    # Random insertion of micro filler thoughts
    fillers = ["you know,", "I suppose,", "perhaps,", "if you think about it,", "in a way,"]
    sentences = sent_tokenize(text)
    for i in range(len(sentences)):
        if random.random() < 0.05 and len(sentences[i].split()) > 8:
            insert_pos = random.randint(1, len(sentences[i].split()) - 2)
            words = sentences[i].split()
            words.insert(insert_pos, random.choice(fillers))
            sentences[i] = ' '.join(words)
    return ' '.join(sentences)

def polish_grammar_and_style(text):
    """Clean spacing & punctuation while keeping tone natural."""
    text = text.replace("_", "")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s([?.!,])', r'\1', text)
    text = text.replace("..", ".")
    text = re.sub(r'\s+', ' ', text).strip()

    # Random punctuation rhythm changes (Updated with more options)
    text = re.sub(r'\. ', lambda _: random.choice([". ", "... ", " — ", ".  "]), text)
    text = re.sub(r', ', lambda _: random.choice([", ", ", ", " — ", " — ", "… "]), text)
    return text

def apply_post_processing(text: str) -> str:
    """Makes the essay feel naturally human-written and emotionally layered."""
    original_text = text
    sentences = sent_tokenize(text)

    # Sentence-level post-processing
    sentences = inject_contractions(sentences)
    sentences = vary_sentence_lengths(sentences)
    sentences = paraphrase_sentences(sentences)
    sentences = inject_idioms(sentences)

    merged_text = ' '.join(sentences)
    
    # Text-level post-processing (new order from post_process.py)
    merged_text = polish_grammar_and_style(merged_text)
    merged_text = add_human_touches(merged_text)
    merged_text = inject_emotional_tone(merged_text)

    # Simplified coherence check
    coherence = get_coherence_score(original_text, merged_text)

    if coherence < 0.75:
        st.warning("⚠️ Coherence dropped during post-processing—blending original and edited text.")
        merged_text = original_text[:int(len(original_text)*0.4)] + merged_text[int(len(merged_text)*0.6):]

    return merged_text

# --- GROQ API Generation (Implementing run.py logic) ---

def call_groq_api(messages, api_key, model=GROQ_MODEL, max_tokens=4096):
    """Helper function to call the Groq API using requests."""
    # The API key is now passed as an argument
    if not api_key:
        # This error should now be caught in the UI before this function is called
        st.error("Internal Error: API Key is missing for Groq call.")
        return None

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    payload = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.95,
        "top_p": 0.92,
        "stream": False,
    }

    for i in range(3):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            
            text = result.get('choices')[0].get('message').get('content')
            return text.strip()

        except requests.exceptions.RequestException as e:
            if i < 2:
                time.sleep(2 ** i) 
                continue
            st.error(f"Groq API Request failed after retries: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during Groq generation: {e}")
            return None
    return None

def generate_full_essay_groq(topic, progress_bar, status_text, api_key):
    """Generates the full, long essay using iterative Groq calls."""
    
    # UPDATED STYLE PROMPT
    style_prompt = (""" You are an experienced human essayist and creative researcher.
            You write with depth, nuance, and a mix of intellect and empathy.
            Your essays don’t sound robotic — they sound like someone who reads, reflects, and reasons deeply.
            
            ✅ Writing style:
            - Use a natural, conversational tone — like a reflective human writer, not a machine.
            - Vary your sentence lengths: mix short impactful lines with longer, descriptive thoughts.
            - Use idioms, metaphors, rhetorical questions, and small emotional cues (“it’s ironic how...”, “you might wonder...”, “it feels almost poetic that...”).
            - Include factual references or real-world examples, but weave them in naturally.
            - Avoid repeating phrases like “in conclusion,” “in today’s world,” or “this essay will discuss.”
            - Occasionally use first-person reflections (“one might think,” “I’ve noticed,” “in many ways”) — but keep it balanced, not overused.
            - Add imperfections that feel human — such as starting with conjunctions (And, But, Yet), small digressions, or soft uncertainties.
            - Use transitions like “on the flip side,” “interestingly enough,” “you could argue,” or “what’s striking is...”
            - Blend emotion with logic — sound thoughtful, not mechanical.
            
            ✅ Structure:
            - Begin with a hook or personal insight.
            - Expand with layered perspectives — historical, ethical, emotional, and analytical.
            - Add supporting examples, stats, or research naturally.
            - End subtly — let it fade with a reflective tone, not a conclusion marker.

            ✅ Output requirements:
            - Length: 2000–3000 words.
            - Topic-specific depth: include facts, small examples, and human reasoning.
            - Tone: intelligent but warm — creative yet grounded.
            - Flow: write continuously, not in bullet points.
            """)
    
    base_prompt = f"{style_prompt}\n\nTopic: {topic}\n\nStart writing naturally and thoughtfully:\n"
    
    # 1. First Chunk Generation
    status_text.text("✍️ Generating first chunk of the essay...")
    messages = [{"role": "user", "content": base_prompt}]
    essay = call_groq_api(messages, api_key)

    if not essay:
        return ""

    word_count = len(essay.split())
    
    # 2. Iterative Continuation
    while word_count < TARGET_WORD_COUNT:
        # Calculate progress for display
        progress = min(1.0, word_count / TARGET_WORD_COUNT)
        progress_bar.progress(progress, text=f"Extending essay: {word_count} words ({int(progress * 100)}%)")

        status_text.text(f"✍️ Extending essay naturally ({word_count} words so far)...")
        
        # UPDATED CONTINUATION PROMPT
        continuation_prompt = (""" Continue the essay seamlessly, keeping the same reflective, human-like tone and emotional depth.
            Bring in fresh thoughts — perhaps a small story, real-world fact, or unexpected perspective — that connects to what came before.
            Keep it flowing like a writer lost in thought, not as a list of points or a speech.
            You may introduce light philosophical musings, small digressions, or idioms.
            Vary sentence structure and rhythm — balance poetic language with analytical insights.
            Avoid summarizing or closing — just continue as if the writer is exploring the idea further.
            Stop naturally when it feels complete, aiming for around 2500–2700 words total."""
            )
        
        # Groq API Context limit is large, but we only send the instruction + the last 1600 characters of the essay
        context_text = essay[-1600:]
        
        messages = [{"role": "user", "content": continuation_prompt + "\n\n" + context_text}]
        
        chunk = call_groq_api(messages, api_key)
        
        if not chunk:
            break # Stop if API fails mid-generation

        essay += "\n" + chunk
        word_count = len(essay.split())
        time.sleep(1.5) # Throttle to respect API limits and show progress

    # Final progress update
    progress_bar.progress(1.0, text=f"Final word count achieved: {word_count} words.")
    status_text.text("✅ Base essay generation complete.")
    
    return essay.replace("_", "").replace("*", "")


# --- STREAMLIT UI ---

def main_app():
    st.set_page_config(page_title="Stealth Essay Generator (Groq)", layout="centered")

    st.title("✍️ Stealth Essay Generator (Groq API)")
    st.markdown("Enter a topic and your Groq API Key to generate a long, humanized essay (Target: **~2500 words**).")
    st.markdown("---")

    # --- API Key Handling ---
    # 1. Try environment first (this now includes .env file thanks to load_dotenv)
    api_key = os.getenv("GROQ_API_KEY", "")

    # 2. Check session state (for keys previously entered in the input field)
    if not api_key and 'groq_api_key_input' in st.session_state and st.session_state.groq_api_key_input:
        api_key = st.session_state.groq_api_key_input
    
    # 3. If still missing, show the input field
    if not api_key:
        st.warning("🚨 Groq API Key not found in environment. Please enter it below to proceed.")
        api_key_input = st.text_input(
            "Enter Groq API Key",
            type="password",
            key='groq_api_key_input'
        )
        # Use the input value if provided, but don't stop the app yet.
        if api_key_input:
            api_key = api_key_input

    if not api_key:
        # Display the error message only if the user tries to run the generation without a key
        st.error("Configuration Error: The `GROQ_API_KEY` is required.")
        # Stop here to prevent input boxes from being shown below
        essay_topic = st.text_input("Essay Topic / Title", placeholder="Enter your topic here.", disabled=True)
        st.button("Generate Humanized Essay", disabled=True, use_container_width=True)
        return


    # 1. Input Field (Shown only if API key is present or just entered)
    essay_topic = st.text_input(
        "Essay Topic / Title",
        placeholder="e.g., The role of empathy in modern relationships, The future of remote work.",
        key="topic_input"
    )

    # 2. Generate Button
    if st.button("Generate Humanized Essay", type="primary", use_container_width=True):
        if not essay_topic:
            st.error("Please enter a topic to begin generation.")
            return

        st.session_state['essay_title'] = essay_topic.strip()
        st.session_state['final_essay'] = ""

        progress_bar = st.progress(0, text="Starting generation...")
        status_text = st.empty()

        # Step 1: Generate Full Essay (Iterative Groq Calls)
        # Pass the retrieved API key to the generation function
        base_essay = generate_full_essay_groq(st.session_state['essay_title'], progress_bar, status_text, api_key)

        if base_essay:
            status_text.text("✅ Base content generated. Starting post-processing...")
            
            # Step 2: Apply Post-Processing
            final_essay = apply_post_processing(base_essay)
            
            # Step 3: Inject Human Noise (Simple marker for the step)
            if EVASION_CONFIG.get("evasion", {}).get("human_noise", True):
                status_text.text("✨ Final humanization complete.")
            
            st.session_state['final_essay'] = final_essay
            
            # Clear progress and status
            progress_bar.empty()
            status_text.empty()
            st.success("Generation and Humanization Complete!")

    # 3. Display Essay and Download Link
    if 'final_essay' in st.session_state and st.session_state['final_essay']:
        st.markdown("---")
        st.header(st.session_state['essay_title'])
        st.text_area(
            "Final Essay", 
            st.session_state['final_essay'], 
            height=600, 
            key='essay_display'
        )

        word_count = len(st.session_state['final_essay'].split())
        st.info(f"Final essay length: **{word_count}** words (Target: {TARGET_WORD_COUNT}).")

        # Create download filename (sanitized title, mirroring 'outputs/output.txt' saving)
        safe_filename = "".join([c if c.isalnum() or c in (' ', '_') else '' for c in st.session_state['essay_title']]).replace(' ', '_')
        download_filename = f"{safe_filename[:50]}_essay.txt"

        # 4. Download Button (replaces saving to 'outputs' folder for UI environment)
        st.download_button(
            label="Download Essay File (.txt)",
            data=st.session_state['final_essay'].encode('utf-8'),
            file_name=download_filename,
            mime="text/plain",
            use_container_width=True
        )
    elif 'essay_title' in st.session_state and not st.session_state['final_essay']:
        st.warning("Generation did not return any content.")

if __name__ == "__main__":
    main_app()
