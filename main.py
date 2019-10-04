from test import *

newest_file = "data/combined_091319.json"

# train_test_split([newest_file], 0.75, 3, True)

n_fold_test([newest_file],10, 3) #AV2E
# n_fold_test([newest_file],10, 2) #AV
# n_fold_test([newest_file],10, 1) #Emotions
# n_fold_test([newest_file],10, 0) #CLARK