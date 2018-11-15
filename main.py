from test import *


data_list = ['./data/game1_big_trial.csv']
print(train_test_split(data_list, 0.75, True))
# print(n_fold_test(data_list,10))