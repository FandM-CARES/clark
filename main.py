from test import *

# data_list = ['data/' + x for x in os.listdir('data')]
# train_test_split(['data/combined_051919_long.json'], 0.75, 2, True)
n_fold_test(['data/combined_051919_long.json'],10, 3) #AV2E
# n_fold_test(['data/combined_051919_long.json'],10, 2) #AV
# n_fold_test(['data/combined_051919_long.json'],10, 1) #Emotions
# n_fold_test(['data/combined_051919_long.json'],10, 0) #CLARK