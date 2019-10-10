from test import n_fold_test, train_test_split

from enums.clark_enums import AV2EClassifiers, NGrams
from enums.global_enums import Models

newest_file = "data/combined_091319.json"

# train_test_split([newest_file], 0.75, 3, True)

# n_fold_test([newest_file],10, Models.APPRAISAL_VARIABLES_TO_EMOTION)
# n_fold_test([newest_file],10, Models.APPRAISAL_VARIABLES)
n_fold_test([newest_file],10, Models.EMOTION, emotion_specifications={"ngram_choice": NGrams.UNIGRAM_AND_BIGRAM.value})
# n_fold_test([newest_file], 10, Models.CLARK, clark_specifications={
#             "av2e_classifier": AV2EClassifiers.RANDOM_FOREST.value,
#             "ngram_choice": NGrams.UNIGRAM_AND_BIGRAM.value
#             })
