import csv
import math
import json

class ProcessedData(object):

    def __init__(self, datasets, c_e):
        self.convos = json.load(open("data/all_convos.json"))
        self.data = self.parse_data(datasets)
        self.c_e = c_e

    def parse_data(self, datasets, file_type="json"):
        """
        Opens fles and parses the data into a multidimensional array

        Parameters:
        datasets (array): array of links to datasets to be parsed

        Returns:
        Array: parsed files
        """
        if file_type == "json":
            return self.parse_json(datasets)
        else:
            return self.parse_csv(datasets)
    
    def parse_json(self, datasets):
        parsed = []
        for dset in datasets:
            with open(dset) as rawData:
                data = json.load(rawData)
                for val in data:
                    for conv in self.convos:
                        if conv["id"] == val["id"]:
                            val["turn1"]["text"] = str(conv["turn1"])
                            val["turn2"]["text"] = str(conv["turn2"])
                            val["turn3"]["text"] = str(conv["turn3"])
                            parsed.append(val)
                            continue
        return parsed
                    
            
    def parse_csv(self, datasets):
        parsed = []
        for dset in datasets:
            with open(dset) as rawData:
                csv_reader = csv.DictReader(rawData)
                for i, row in enumerate(csv_reader):
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
