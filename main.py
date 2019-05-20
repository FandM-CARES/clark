from test import *

# data_list = ['data/' + x for x in os.listdir('data')]
# train_test_split(['data/combined_051919.json'], 0.75, 0, True)
# n_fold_test(['data/final_emotions_051819.json'],10, 1)
n_fold_test(['data/combined_051919_long.json'],10, 0)