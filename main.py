from test import *
import os
import csv

with open('data/train.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("\t") for line in stripped if line)
    with open('data/train.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)

with open('data/testwithoutlabels.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("\t") for line in stripped if line)
    with open('data/testwithoutlabels.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)

#train_test_split(['data/train.csv'], 0.85, 1, False)
#n_fold_test(['data/train.csv'],10, 1)
run_full_test(['data/train.csv'], ['data/testwithoutlabels.csv'])