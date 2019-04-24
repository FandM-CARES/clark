import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import math

class ProcessedData(object):

    def __init__(self, datasets, split=None):
        self.data = self.parse_data(datasets, split)

    def parse_data(self, datasets, split):
        """
        Opens CSV fles and parses the data into a multidimensional array

        Parameters:
        datasets (array): array of links to datasets to be parsed

        Returns:
        Array: parsed CSV files
        """

        if split != None:
            amt_angry = 0.183 * split
            amt_happy = 0.141 * split
            amt_others = 0.495 * split
            amt_sad = 0.181 * split

        parsed = []
        for dset in datasets:
            with open(dset) as rawData:
                csv_reader = csv.DictReader(rawData)
                for i, row in enumerate(csv_reader):
                    if split != None:
                        if row['label']  == 'angry' and amt_angry > 0:
                            amt_angry -= 1
                            parsed.append(row)
                        if row['label'] == 'sad' and amt_sad > 0:
                            amt_sad -= 1
                            parsed.append(row)
                        if row['label'] == 'happy' and amt_happy > 0:
                            amt_happy -= 1
                            parsed.append(row)
                        if row['label'] == 'others' and amt_others > 0:
                            amt_others -= 1
                            parsed.append(row)
                    else:
                        parsed.append(row)
                    
        return parsed

    def split_data(self, datasets, split=0.75):
        """
        Splits data into training and testing sets 

        Parameters:
        datasets (array): collection of all data
        split (float): percent split training to testing, default is .75

        Returns:
        Array: training data
        Array: testing data
        """
        train = []
        test = []
        for dset in datasets:
            amount_of_training = math.floor(split * sum(1 for row in dset))
            for i, row in enumerate(dset):
                if i == 0:
                    continue
                if i > amount_of_training:
                    test.append(row)
                else:
                    train.append(row)
        return train, test
