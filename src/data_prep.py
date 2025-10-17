import json
import random
from datasets import Dataset  
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt',quiet=True)

def load_dataset(file_path: str, augment: bool = True)->Dataset:
    """Load JSON dataset of texts. Optionally augment for burstiness training."""
    with open(file_path,'r') as f:
        data = json.load(f)
    
    if augment:
        #simple augmentation: vary sentence order randomly for diversity
        for item in data:
            sentences = sent_tokenize(item['text'])
            random.shuffle(sentences[:len(sentences)//2])
            item['text'] = ' '.join(sentences)

    return Dataset.from_list([{'text': d['text']} for d in data])

#example usage(for testing)
if __name__ == '__main__':
    #create dummy data
    dummy = [{'text': 'This is a sample sentence. Another one here.'}]
    with open('../data/dummy.json','w') as f:
        json.dump(dummy, f)
    ds = load_dataset('../data/dummy.json')
    print(ds)

