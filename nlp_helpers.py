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
    contractions_dict = {
        "are not": "aren't",
        "cannot": "can't",
        "can not": "can't",
        "could have": "could've",
        "did not": "didn't",
        "does not": "doesn't",
        "do not": "don't",
        "going to": "gonna",
        "have not": "haven't",
        "he would": "he'd",
        "he has": "he's",
        "he is": "he's",
        "how did": "how'd",
        "how are": "how're",
        "how is": "how's",
        "i would": "i'd",
        "i will": "i'll",
        "i am ": "i'm",
        "i have": "i've",
        "is not": "isn't",
        "it is": "it's",
        "let us": "let's",
        "she would": "she'd",
        "she has": "she's",
        "she is": "she's",
        "someone is": "someone's",
        "that is": "that's",
        "there is": "there's",
        "we would": "we'd",
        "we are": "we're",
        "we have": "we've",
        "what is": "what's",
        "when is": "when's",
        "where is": "where's",
        "who would": "who'd",
        "will not": "won't",
        "you would": "you'd",
        "you are": "you're",
        "you have": "you've"
    }

    init_res = nltk.word_tokenize(row)
    for i, word in enumerate(init_res):
        init_res[i] = word.lower()
        if i+1 < len(init_res):
            pot_contraction = init_res[i] + " " + init_res[i+1]
            if (pot_contraction in contractions_dict):
                init_res[i] = contractions_dict[pot_contraction]
                del init_res[i+1]
                continue
        if word[0] == "'":
            init_res[i-1] = init_res[i-1] + init_res[i]
            del init_res[i]
            continue
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
    