import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer,util
import numpy as np
import logging

nltk.download('wordnet',quiet=True)
nltk.download('punkt',quiet=True)

syn_model = SentenceTransformer('all-MiniLM-L6-v2')
logger  = logging.getLogger(__name__)

def apply_post_processing(text: str, config: dict) -> str:
    """
    Apply steps: vary lengths, paraphrase, inject idioms.
    Ensures coherence >0.9 similarity.
    """
    steps = config.get('rules', {}).get('post_process_steps', ['vary_lengths', 'paraphrase', 'inject_idioms'])
    
    sentences = sent_tokenize(text)
    
    for step in steps:
        if step == 'vary_lengths':
            sentences = vary_sentence_lengths(sentences, config.get('rules', {}).get('sentence_var', {}))
        elif step == 'paraphrase':
            sentences = paraphrase_sentences(sentences)
        elif step == 'inject_idioms':
            sentences = inject_idioms(sentences, config.get('rules', {}).get('idioms', []))
    
    # Additional burstiness boost: Shuffle 20% sentences
    if random.random() < 0.2:
        random.shuffle(sentences[:len(sentences)//5])  # Shuffle 20%
    
    new_text = ' '.join(sentences)
    orig_emb = syn_model.encode([text])
    new_emb = syn_model.encode([new_text])
    similarity = util.cos_sim(orig_emb, new_emb).item()
    if similarity < 0.9:
        logger.warning("Coherence low; skipping heavy edits")
        return text  # Fallback
    
    return new_text

def vary_sentence_lengths(sentences, var_config):
    lens = [len(s.split()) for s in sentences]
    std = np.std(lens)
    target_std = 0.9 * np.mean(lens)  # Burstiness target
    
    for i in range(len(sentences)):
        if std < target_std:
            words = sentences[i].split()
            if random.random() < 0.5 and len(words) > var_config.get('min_len', 5):
                # Shorten: Remove 1-3 words
                del words[random.randint(0, min(3, len(words)//2))]
                sentences[i] = ' '.join(words) + '.' if not sentences[i].endswith('.') else ' '.join(words)
            else:
                # Lengthen: Add filler words/idioms
                fillers = ['however', 'moreover', 'in fact', 'on the other hand']
                sentences[i] += ' ' + random.choice(fillers)
        std = np.std([len(s.split()) for s in sentences])
    return sentences

def paraphrase_sentences(sentences):
    def paraphrase_sent(sent):
        words = sent.split()
        synonym_map = config.get('rules', {}).get('synonym_map', {})
        for i, word in enumerate(words):
            if word.lower() in synonym_map and random.random() < 0.3:
                words[i] = random.choice(synonym_map[word.lower()])
        return ' '.join(words)
    return [paraphrase_sent(s) for s in sentences]

def inject_idioms(sentences, idioms):
    if random.random() < 0.2:
        idx = random.randint(0, len(sentences)-1)
        idiom = random.choice(idioms)
        sentences[idx] += f" Like {idiom}."

    return sentences
