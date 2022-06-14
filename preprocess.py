import os
import re  # For Preprocessing
from string import punctuation  # For Preprocessing
import nltk
from nltk.corpus import stopwords  # For Preprocessing


def preprocess_data(data):
    """
    Function removes Stopwords, Digits, Punctuations, Lowercasts and takes <list> as Input.
    """

    german_stop_words = stopwords.words('german')
    german_stop_words.extend(["wer", "darauf", "sowohl", "allerdings", "sowie", "zT", "erst", "daher", "zB"])
    cleaned_data = " ".join([w for w in data if not w.isalpha()])  # Remove Digits
    cleaned_data = re.sub(f"[{re.escape(punctuation)}]", "", cleaned_data).lower()  # Remove Punctuations & Lower
    cleaned_data = " ".join([s for s in cleaned_data.split() if not s in german_stop_words])
    data_tokens = nltk.word_tokenize(cleaned_data)

    return [cleaned_data, data_tokens]
