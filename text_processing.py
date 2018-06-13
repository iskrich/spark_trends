from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from stop_words import *
import string
from pymystem3 import Mystem
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
nltk.download('punkt')

rus_stem = Mystem()
en_stem = EnglishStemmer()


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def stem(text):
    return [word for word in rus_stem.lemmatize(text)
             if not set(string.punctuation) & set(word)
             and word.strip()
             and word not in stopwords.words('russian').extend('https', 'ru')]