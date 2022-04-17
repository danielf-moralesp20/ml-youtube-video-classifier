import re
import string
from langdetect import detect_langs
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


def is_english(text: str, threshold_confidence: float = 0.95) -> bool:
    detected_langs = detect_langs(text)
    en_lang = next(
        (lang for lang in detected_langs if lang.lang == 'en'), None)
    return en_lang != None and en_lang.prob >= threshold_confidence


def clean_text(text: str, min_token_length: int, lang: str = 'english') -> str:
    ss = SnowballStemmer(language=lang)

    text = re.sub(f'[{string.punctuation}]+', '', text)  # Removing Puntuations
    document = TweetTokenizer(preserve_case=False, reduce_len=True).tokenize(
        text)  # Tokenization and Case Normalization
    document = [token for token in document if token not in stopwords.words(
        lang) and len(token) >= min_token_length]  # Removing Stopwords
    document = [ss.stem(token) for token in document]  # Stemming

    return document
