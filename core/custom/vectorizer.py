import logging
import numpy as np
from ..utils import example_text

logger = logging.getLogger(__name__)

class MVectorizer:
    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None
        
    def build_vocab(self, texts: list[str]):
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            for word in words:
                unique_words.add(word)
            
        self.word2idx = {word: idx + 1 for idx, word in enumerate(sorted(unique_words))}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        logger.info(f'word2idx: {self.word2idx}\n idx2word: {self.idx2word}')
        
        vocab_size = len(self.word2idx) + 1
        self.embeddings = np.random.randn(vocab_size, self.embedding_dim)
        logger.info(f'self.embeddings: {self.embeddings}')
        
    def text_to_sequence(self, text: str):
        words = text.lower().split()
        return [self.word2idx.get(word, 0) for word in words]
    
    def sequence_to_embeddings(self, sequence):
        return np.array([self.embeddings[idx] for idx in sequence])
    
    def transform(self, text: str):
        sequence = self.text_to_sequence(text)
        return self.sequence_to_embeddings(sequence)
    
def vect_use(embedding_dim: int = 8, texts: list[str] = example_text):
    vectorizer = MVectorizer(embedding_dim)
    vectorizer.build_vocab(texts)
    
    logger.info(f'Словарь: {vectorizer.word2idx}')
    return vectorizer
         
        