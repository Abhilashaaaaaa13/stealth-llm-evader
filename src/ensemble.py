import random
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt',quiet=True)

def ensemble_merge(drafts: list[str])->str:
    """Merge drafts: Sentence shuffle + unique."""
    all_sentences = []
    for draft in drafts:
        all_sentences.extend(sent_tokenize(draft))

    #remove duplicates(keep diverse)
    unique_sents = list(set(all_sentences))
    random.shuffle(unique_sents)  #boost perplexity

    #rebuild to target length
    target_sents = unique_sents[:len(all_sentences)]
    return ' '.join(target_sents)