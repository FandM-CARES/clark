import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

"""
Helper functions for parsing inputs
"""

def tokenize(row):
    """
    Tokenizes the row exluding .;,:/-_&~

    Parameters:
    row (string): row of data

    Returns:
    String: tokenized word
    """

    bad_characters = ['.',';',':','/','-','_','&','~',',', '\\']
    init_res = nltk.word_tokenize(row)
    for i, word in enumerate(init_res):
        init_res[i] = word.lower()
        if word[0] == "'":
            init_res[i-1] = init_res[i-1] + init_res[i]
            del init_res[i]
        if word in bad_characters:
            del init_res[i]         
    
    return init_res

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
    