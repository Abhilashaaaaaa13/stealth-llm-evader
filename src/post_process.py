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

def apply_post_processing(text:str, config:dict)->str:
    """Apply steps:vary lengths,paraphrase, inject idioms.
    ensures coherence >0.9 similarity."""
    steps = config.get('rules',{}).get('post_process_steps',['vary_lengths','paraphrase','inject_idioms'])

    sentences = sent_tokenize(text)

    for step in steps:
        if step == 'vary_lengths':
            sentences = vary_sentence_lengths(sentences, config.get('rules',{}).get('sentence_var',{}))
        
        elif step == 'paraphrase':
            sentences = paraphrase_sentences(sentences)

        elif step == 'inject_idioms':
            sentences = inject_idioms(sentences, config.get('rules',{}).get('idioms',[]))

    #coherence check : rebuild and verify similarity to original
    new_text = ' '.join(sentences)
    orig_emb = syn_model.encode([text])
    new_emb = syn_model.encode([new_text])
    similarity = util.cos_sim(orig_emb,new_emb).item()
    if similarity < 0.9:
        logger.warning("Coherence low; skipping heavy edits")
        return text
    
    return new_text

def vary_sentence_lengths(sentences, var_config):
    lens = [len(s.split()) for s in sentences]
    std = np.std(lens)
    target_std = 0.9*np.mean(lens) 

    for i in range(len(sentences)):
        if std < target_std:
            #shorten or lengthen randomly
            if random.random() < 0.5 and len(sentences[i].split()) > var_config.get('min_len',5):
                #remove words
                words = sentences[i].split()
                del words[random.randint(0,len(words)//2)]
                sentences[i] = ' '.join(words)
            
            else:
                #add filler(simple)
                sentences[i] += 'indeed.'
        std = np.std([len(s.split()) for s in sentences])
    return sentences

def paraphrase_sentences(sentences):
    def paraphrase_sent(sent):
        words = sent.split()
        if words:
            # Replace 1-2 words with synonyms
            for i, word in enumerate(words[:2]):
                for syn in wordnet.synsets(word):
                    if syn.lemmas():
                        new_word = syn.lemmas()[0].name()
                        words[i] = new_word
                        break
        return ' '.join(words)
    return [paraphrase_sent(s) for s in sentences]

def inject_idioms(sentences, idioms):
    if random.random() < 0.2:
        idx = random.randint(0, len(sentences)-1)
        idiom = random.choice(idioms)
        sentences[idx] += f" Like {idiom}."

    return sentences
