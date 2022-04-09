import re
import string
from langdetect import detect_langs
from nltk.corpus import stopwords


def is_english(text: str, threshold_confidence: float = 0.95) -> bool:
    detected_langs = detect_langs(text)
    en_lang = next(
        (lang for lang in detected_langs if lang.lang == 'en'), None)
    return en_lang != None and en_lang.prob >= threshold_confidence


def drop_punctuation(text: str) -> str:
    return re.sub(f'[{string.punctuation}]+', '',  text)


def drop_stopwords(text: str, lang: str = 'english') -> str:
    return ' '.join([word for word in tokenize(text) if word not in stopwords.words(lang)])


def tokenize(text):
    return re.split('\W+', text.lower())
