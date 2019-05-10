import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

"""
Helper functions for parsing inputs
"""

def flatten(l):
    return [item for sublist in l for item in sublist]

def tokenize(row, ngram=0):
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

    bad_characters = ['.','-','_','&','~',',','\\']
    
    if row == 24:
        print(hi)
    init_res = nltk.tokenize.casual.casual_tokenize(row)
    for i, word in enumerate(init_res):
        init_res[i] = word.lower()
        if word in bad_characters:
            init_res[i] = ""
        if ngram == 0:
            if is_stop_word(word):
                init_res[i] = ""
    
    temp_ret = [x for x in init_res if x != ""]
    
    if len(temp_ret) == 0:
        return [], ""

    if ngram == 0: 
        return temp_ret, row
    
    if ngram == 1: 
        return [" ".join(x) for x in list(nltk.bigrams(temp_ret))], row  
    
    if ngram == 2:
        return [x for x in temp_ret if not is_stop_word(x)] + [" ".join(x) for x in list(nltk.bigrams(temp_ret))], row

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
    