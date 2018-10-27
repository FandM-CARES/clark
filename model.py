from processedData import *
import math
# from sklearn.model_selection import KFold
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

class Model(object):

    variable = ['Pleasantness', 'Attention', 'Control',
                'Certainty', 'Anticipated Effort', 'Responsibililty']

    def __init__(self, datasets, train_test_split):
        self.datasets = datasets
        self.trained = False
        self.accuracy = 0
        self.trainingData = None
        self.labelledTestingData = None
        self.classification = self.processData(datasets, train_test_split)

    def test(self):
        correct = 0
        total = len(self.labelledTestingData)
        accuracies = {}

        for var in self.variable:
            accuracies[var] = 0

        for message in self.labelledTestingData:
            for var in self.variable:
                parsed_message = ' '.join([word for word in message.split() if word not in stop_words])
                classification = self.classify(
                    self.trainingData.unigrams[var], parsed_message, self.trainingData.totals[var], self.trainingData.PPs[var])
                if classification == self.labelledTestingData[message][var]:
                    accuracies[var] += 1

        for var in accuracies:
            accuracies[var] = float(accuracies[var]/total)

        self.accuracy = accuracies

    def processData(self, datasets, train_test_split):
        train, test = self.split_data(datasets, train_test_split)

        self.labelledTestingData = ProcessedData(
            test, self.variable).labelledData

        for dset in datasets:
            self.trainingData = ProcessedData(train, self.variable)
            normalizedDict = self.normalizeValues(
                self.trainingData.unigrams, self.trainingData.totals)

        self.test()

        return {}

    def split_data(self, datasets, split):
        train = []
        test = []
        for dset in datasets:
            with open(dset) as rawData:
                csv_reader = csv.DictReader(rawData)
                amount_of_training = math.floor(
                    split * sum(1 for row in csv_reader))
                # resetting the iterator to the start of the file
                rawData.seek(0)
                for i, row in enumerate(csv_reader):
                    if i == 0:
                        continue
                    if i > amount_of_training:
                        test.append(row)
                    else:
                        train.append(row)
        return train, test

    def normalizeValues(self, labelledUnigrams, totals):
        for var in labelledUnigrams:
            for word in labelledUnigrams[var]:
                if labelledUnigrams[var][word]['low'] == 0:
                    labelledUnigrams[var][word]['low'] = float(
                        1)/float(totals[var]['num_low']+1)
                else:
                    labelledUnigrams[var][word]['low'] = float(
                        labelledUnigrams[var][word]['low'])/float(totals[var]['num_low'])
                if labelledUnigrams[var][word]['med'] == 0:
                    labelledUnigrams[var][word]['med'] = float(
                        1)/float(totals[var]['num_med']+1)
                else:
                    labelledUnigrams[var][word]['med'] = float(
                        labelledUnigrams[var][word]['med'])/float(totals[var]['num_med'])
                if labelledUnigrams[var][word]['high'] == 0:
                    labelledUnigrams[var][word]['high'] = float(
                        1)/float(totals[var]['num_high']+1)
                else:
                    labelledUnigrams[var][word]['high'] = float(
                        labelledUnigrams[var][word]['high'])/float(totals[var]['num_high'])

        return labelledUnigrams

    def classify(self, trainingDict, content, totals, PPs):
        sumLow = float(0)
        sumMed = float(0)
        sumHigh = float(0)

        for word in content:
            if word in trainingDict:
                sumLow += float(math.log(trainingDict[word]['low'], 10))
                sumMed += float(math.log(trainingDict[word]['med'], 10))
                sumHigh += float(math.log(trainingDict[word]['high'], 10))

        lowProb = math.log(PPs['low']) + sumLow
        medProb = math.log(PPs['med']) + sumMed
        highProb = math.log(PPs['high']) + sumHigh

        maxVal = max([lowProb, medProb, highProb])
        if lowProb == maxVal:
            return 'low'
        if medProb == maxVal:
            return 'med'
        else:
            return 'high'
