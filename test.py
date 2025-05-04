from core.custom.vectorizer import MVectorizer, vect_use
import logging

logger = logging.getLogger(__name__)

text = 'купить промокод сегодня'

vectorizer = vect_use(8)
vectors = vectorizer.transform(text)

print("Индексы:", vectorizer.text_to_sequence(text))
print("Вектора:\n", vectors)
