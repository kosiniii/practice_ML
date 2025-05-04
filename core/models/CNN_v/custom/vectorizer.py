import json
import logging
import numpy as np
from ....utils import example_text
import re

logger = logging.getLogger(__name__)

class MVectorizer:
    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None
        
    def clean_text(self, text: str) -> str:
        text = text.lower()
        cleaned = re.sub('r[^а-яa-z0-9\s]', '', text)
        return cleaned
    
    def build_vocab(self, texts: list[str]):
        unique_words = set()
        for text in texts:
            words = self.clean_text(text).split()
            for word in words:
                unique_words.add(word)
        
        self.word2idx = {'UNK': 1, 'PAD': 0}
        for idx, word in enumerate(sorted(unique_words), start=2):
            self.word2idx[word] = idx

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f'word2idx: {self.word2idx}\n idx2word: {self.idx2word}')
        
        vocab_size = len(self.word2idx)
        self.embeddings = np.random.normal(0, 1, (vocab_size, self.embedding_dim))
        print(f'self.embeddings: {self.embeddings}')
    
    def sequence_to_embeddings(self, sequence):
        return np.array([self.embeddings[idx] for idx in sequence])
    
    def transform(self, text: str):
        sequence = self.text_to_sequence(text)
        return self.sequence_to_embeddings(sequence)

    def padding(self, sequence, max_len: int = 10):
        if len(sequence) < max_len:
            sequence += [self.word2idx['PAD']] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]
        return sequence
    
    def transform_batch_texts(self, texts: list[str], max_len: int = 30):
        sequences = [self.text_to_sequence(text) for text in texts]
        padded_sequences = [self.padding(seq, max_len) for seq in sequences]
        return np.array(padded_sequences)
        
    def save_vocab(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False)
            

    def load_vocab(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    
    def text_to_sequence(self, text: list[str]):
        word = self.clean_text(text).split()
        if not word:
            return [self.word2idx['PAD']]
        else:
            return [self.word2idx.get(word, self.word2idx['UNK']) for word in word]
    
    
def vect_use(embedding_dim: int = 8, texts: list[str] = example_text):
    vectorizer = MVectorizer(embedding_dim)
    vectorizer.build_vocab(texts)
    
    print(f'Словарь: {vectorizer.word2idx}')
    return vectorizer
         
        