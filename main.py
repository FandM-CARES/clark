from test import n_fold_test

from utils.enums.clark_enums import AV2EClassifiers
from utils.enums.global_enums import Models, NGrams

newest_file = "data/combined_091319_all_1.json"

# n_fold_test([newest_file], 10, Models.APPRAISAL_VARIABLES_TO_EMOTION)
# n_fold_test([newest_file], 10, Models.APPRAISAL_VARIABLES)
# n_fold_test([newest_file], 10, Models.EMOTION)
n_fold_test([newest_file], 10, Models.CLARK)
