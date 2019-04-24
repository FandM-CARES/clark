from baseline import *
import csv
import numpy as np

clark = list()
truth = list()

emotion2encoding = {
    'others': [1,0,0,0],
    'happy': [0,1,0,0],
    'sad': [0,0,1,0],
    'angry': [0,0,0,1]
}

def main():

    with open('test_clark.txt') as clark_data:
        csv_reader = csv.DictReader(clark_data, delimiter = '\t')
        for row in csv_reader:
            clark.append(emotion2encoding[row['label']])

    with open('test.txt') as truth_data:
        csv_reader = csv.DictReader(truth_data, delimiter = '\t')
        for row in csv_reader:
            truth.append(emotion2encoding[row['label']])

    getMetrics(np.array(clark), np.array(truth))

if __name__ == "__main__":
    main()