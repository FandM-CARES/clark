from test import *

data_list = ['./data/game1_big_trial.csv', './data/game2_big_trial_2.csv'] # convert this to just take everything that's in the data folder
# print(train_test_split(data_list, 0.85, True))
print(n_fold_test(data_list,10))