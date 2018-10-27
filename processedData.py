import csv
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import random

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


class ProcessedData(object):

    dataPoints = {}
    totals = {}
    PPs = {}
    bounds = {}

    def __init__(self, trainingDset, variables):
        # self.input = linkToRawDataSet
        self.unigrams = self.processDataUnigrams(trainingDset, variables)
        self.labelledData = self.labelData(trainingDset, variables)
    
    def labelData(self, dset, vars):
        labelledData = {}
        for row in dset:
            message = re.sub(r'[.!,;?]', "",  row['Player Message'])
            labelledData[message] = {}
            for var in vars:
                weight = self.numToClassificationWeight(row, var)
                labelledData[message][var] = weight
        return labelledData

    def processDataUnigrams(self, linkToRawDataset, variables):
        result = {}
        for var in variables:
            unigrams, totals, PPs = self.parseCSVIntoLabelledUnigrams(
                linkToRawDataset, var)
            self.totals[var] = totals
            self.PPs[var] = PPs
            result[var] = unigrams
        return result

    def parseCSVIntoLabelledUnigrams(self, trainingDSet, variable, words={}):
        totals = {
            'num_low': 0,
            'num_med': 0,
            'num_high': 0
        }

        self.dataPoints[variable] = []

        # with open(linkToRawDataset, mode='r') as rawData:
        #     csv_reader = csv.DictReader(rawData)
        for row in trainingDSet:
            weight = self.numToClassificationWeight(row, variable)
            self.dataPoints[variable].append(float(row[variable]))
            # substituting any non-alphanumerical with a space and then splitting on it to get words
            for word in re.sub(r'[.!,;?]', " ",  row['Player Message']).split():
                if word in stop_words:
                    continue
                word = word.lower()
                if word in words:
                    words[word][weight] += 1
                    totals = self.allocateClassificationWeightToTotal(
                        weight, totals)
                else:
                    words[word] = {'low': 0, 'med': 0, 'high': 0}
                    words[word][weight] = 1
                    totals = self.allocateClassificationWeightToTotal(
                        weight, totals)

        PPs = {
            'low': float(totals['num_low'])/float(totals['num_low'] + totals['num_med'] + totals['num_high']),
            'med': float(totals['num_med'])/float(totals['num_low'] + totals['num_med'] + totals['num_high']),
            'high': float(totals['num_high'])/float(totals['num_low'] + totals['num_med'] + totals['num_high'])
        }

        return words, totals, PPs

    def plotData(self):
        colors = ["red", "blue", "green", "yellow", "brown", "purple"]
        i = 0
        for var in self.dataPoints:
            sns.distplot(self.dataPoints[var], color=colors[i], label=var)
            i += 1
        plt.legend()
        plt.show()

    def calcSTDData(self):
        res = []
        for var in self.dataPoints:
            res.append([np.mean(self.dataPoints[var]),
                        np.std(self.dataPoints[var]), var])
        return res

    def calcBoundsData(self, stdData):
        for var in stdData:
            lb = var[0] - (var[1] / 2)
            ub = var[0] + (var[1] / 2)
            print(lb, ub, var[2])

    def numToClassificationWeight(self, row, variable):
        def numToCWForPleasantness(row):
            if float(row['Pleasantness']) < 0.817:
                return 'low'
            if float(row['Pleasantness']) < 1.214:
                return 'med'
            return 'high'

        def numToCWForAttention(row):
            if float(row['Attention']) < 1.007:
                return 'low'
            if float(row['Attention']) < 1.430:
                return 'med'
            return 'high'

        def numToCWForControl(row):
            if float(row['Control']) < 0.963:
                return 'low'
            if float(row['Control']) < 1.410:
                return 'med'
            return 'high'

        def numToCWForCertainty(row):
            if float(row['Certainty']) < 0.915:
                return 'low'
            if float(row['Certainty']) < 1.341:
                return 'med'
            return 'high'

        def numToCWForAnticipatedEffort(row):
            if float(row['Anticipated Effort']) < 0.960:
                return 'low'
            if float(row['Anticipated Effort']) < 1.366:
                return 'med'
            return 'high'

        def numToCWForResponsibility(row):
            if float(row['Responsibililty']) < 0.913:
                return 'low'
            if float(row['Responsibililty']) < 1.291:
                return 'med'
            return 'high'

        if variable == 'Pleasantness':
            return numToCWForPleasantness(row)
        if variable == 'Attention':
            return numToCWForAttention(row)
        if variable == 'Control':
            return numToCWForControl(row)
        if variable == 'Certainty':
            return numToCWForCertainty(row)
        if variable == 'Anticipated Effort':
            return numToCWForAnticipatedEffort(row)
        if variable == 'Responsibililty':
            return numToCWForResponsibility(row)

    def allocateClassificationWeightToTotal(self, cw, totals):
        if cw == 'low':
            totals['num_low'] += 1
        elif cw == 'med':
            totals['num_med'] += 1
        elif cw == 'high':
            totals['num_high'] += 1
        return totals
