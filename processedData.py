import csv
import re
import nltk

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

class ProcessedData(object):

    def __init__(self, linkToRawDataSet, variables):
        self.input = linkToRawDataSet
        self.result = self.processDataUnigrams(linkToRawDataSet, variables)

    def processDataUnigrams(self, linkToRawDataset, variables):
        result = {}
        for var in variables:
            unigrams, totals, PPs = self.parseCSVIntoLabelledUnigrams(linkToRawDataset, var)
            result[var] = self.normalizeValues(unigrams, totals)
        return result

    def parseCSVIntoLabelledUnigrams(self, linkToRawDataset, variable, words={}):
        totals = {
            'num_low': 0,
            'num_med': 0,
            'num_high': 0
        }

        with open(linkToRawDataset, mode='r') as rawData:
            csv_reader = csv.DictReader(rawData)
            for row in csv_reader:
                weight = self.numToClassificationWeight(row, variable)
                for word in re.sub(r'[.!,;?]', " ",  row['Player Message']).split(): # substituting any non-alphanumerical with a space and then splitting on it to get words
                    if word in stop_words:
                        continue
                    word = word.lower()
                    if word in words:
                        words[word][weight] += 1
                        totals = self.allocateClassificationWeightToTotal(weight, totals)
                    else:
                        words[word] =  {'low': 0, 'med': 0, 'high': 0} 
                        words[word][weight] = 1
                        totals = self.allocateClassificationWeightToTotal(weight, totals)

        PPs = {
            'low': float(totals['num_low'])/float(totals['num_low'] + totals['num_med'] + totals['num_high'])
            'med': float(totals['num_med'])/float(totals['num_low'] + totals['num_med'] + totals['num_high'])
            'high': float(totals['num_high'])/float(totals['num_low'] + totals['num_med'] + totals['num_high'])
        }

        return words, totals, PPs

    def numToClassificationWeight(self, row, variable):
        if float(row[variable]) < 1.0: 
            return 'low'
        elif float(row[variable]) < 2.0:
            return 'med'
        else: 
            return 'high'
    
    def allocateClassificationWeightToTotal(self, cw, totals):
        if cw == 'low':
            totals['num_low'] += 1
        elif cw == 'med':
            totals['num_med'] += 1
        elif cw == 'high':
            totals['num_high'] += 1
        return totals

    def normalizeValues(self, labelledUnigrams, totals):
        for word in labelledUnigrams:
            if labelledUnigrams[word]['low'] == 0:
                labelledUnigrams[word]['low'] = float(1)/float(totals['num_low']+1)
            else:
                labelledUnigrams[word]['low'] = float(labelledUnigrams[word]['low'])/float(totals['num_low'])
            if labelledUnigrams[word]['med'] == 0:
                labelledUnigrams[word]['med'] = float(1)/float(totals['num_med']+1)
            else:
                labelledUnigrams[word]['med'] = float(labelledUnigrams[word]['med'])/float(totals['num_med'])
            if labelledUnigrams[word]['high'] == 0:
                labelledUnigrams[word]['high'] = float(1)/float(totals['num_high']+1)
            else:
                labelledUnigrams[word]['high'] = float(labelledUnigrams[word]['high'])/float(totals['num_high'])
        
        return labelledUnigrams