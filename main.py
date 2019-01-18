from test import *
import os

data_list = ['data/' + x for x in os.listdir('data')]
train_test_split(data_list, 0.85, 0, True)
#n_fold_test(data_list,10, 0)