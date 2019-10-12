from enum import Enum

class NGrams(Enum):
    UNIGRAM = 0
    BIGRAM = 1
    UNIGRAM_AND_BIGRAM = 2

class Models(Enum):
    APPRAISAL_VARIABLES = 0
    APPRAISAL_VARIABLES_TO_EMOTION = 1
    CLARK = 2
    EMOTION = 3