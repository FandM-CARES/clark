from test import *
import os

# data_list = ['data/' + x for x in os.listdir('data')]
# train_test_split(['data/final_appraisals_050819.json'], 0.85, 0, True)
n_fold_test(['data/final_emotions_050819.json'],10, 1)
# n_fold_test(['data/final_appraisals_050919.json'],10, 0)