import nltk
import numpy as np
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

"""
Helper functions for parsing inputs
"""


def normalize(arr):
    """
    Normalizes between 0.1 and 1.0
    """
    a = 0.9 * (arr - np.min(arr))/np.ptp(arr) + 0.1
    return a/a.sum(0)


def flatten(l):
    return [item for sublist in l for item in sublist]


def determine_tense(tagged_tuple):
    tag = tagged_tuple[1]
    if tag in ["VBG", "VBP", "VBZ"]:
        return "present"
    elif tag in ["VBD", "VBN"]:
        return "past"
    elif tag in ["MD"]:
        return "future"
    return ""


def determine_pronoun(tagged_tuple):
    tag = tagged_tuple[1]
    if tag in ["WP", "WP$", "PRP", "PRP$"]:
        obj = tagged_tuple[0].lower()
        if obj in ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"]:
            return "first"
        elif obj in ["you", "your", "yours", "yourself", "yourselves"]:
            return "second"
        elif obj in ["he", "his", "him", "himself", "her", "hers", "she", "herself", "they", "them", "their", "theirs", "themselves", "it", "its", "itself"]:
            return "third"
        else:
            return ""
    else:
        return ""


def parts_of_speech(tokenized_res):
    return nltk.pos_tag(tokenized_res)


def tokenize(row):
    return nltk.tokenize.casual.casual_tokenize(row)


def ngrams_and_remove_stop_words(init_res, ngram):
    """
    Tokenizes the row exluding .;,:/-_&~ and removes stop words

    Parameters:
    row (string): row of data
    ngram (integer): specifies which kind(s) of ngrams to use
    - 0 : unigrams
    - 1 : bigrams
    - 2 : unigrams and bigrams

    Returns:
    String: tokenized word
    """

    res = list(init_res)

    bad_characters = [".", "-", "_", "&", "~", ",", "\\"]

    for i, word in enumerate(res):
        res[i] = word.lower()
        if word in bad_characters:
            res[i] = ""
        if ngram == 0 and is_stop_word(word):
            res[i] = ""

    temp_ret = [x for x in res if x != ""]

    if len(temp_ret) == 0:
        return []

    if ngram == 0:
        return temp_ret

    if ngram == 1:
        return [" ".join(x) for x in list(nltk.bigrams(temp_ret))]

    if ngram == 2:
        return [x for x in temp_ret if not is_stop_word(x)] + [" ".join(x) for x in list(nltk.bigrams(temp_ret))]


def is_stop_word(word):
    """
    Determines if a word is classified as a stop word

    Parameters:
    word (string): potential stop word

    Returns:
    Bool: True if stop word
    """
    if word in stop_words:
        return True
    return False
