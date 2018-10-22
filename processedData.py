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
            result[var] = self.normalizeValues(self.parseCSVIntoLabelledUnigrams(linkToRawDataset, var))
        return result

    def parseCSVIntoLabelledUnigrams(self, linkToRawDataset, variable, words={}):
        with open(linkToRawDataset, mode='r') as rawData:
            csv_reader = csv.DictReader(rawData)
            for row in csv_reader:
                weight = self.determineWeight(row, variable)
                for word in re.sub(r'[.!,;?]', " ",  row['Player Message']).split(): # substituting any non-alphanumerical with a space and then splitting on it to get words
                    if word in stop_words:
                        continue
                    word = word.lower()
                    if word in words:
                        words[word][weight] += 1
                    else:
                        words[word] =  {'low': 0, 'med': 0, 'high': 0} 
                        words[word][weight] = 1
                        
        return words 

    def determineWeight(self, row, variable):
        if float(row[variable]) < 1.0: 
            return 'low'
        elif float(row[variable]) < 2.0:
            return 'med'
        else: 
            return 'high'
    
    def normalizeValues(labelledUnigrams):
        for word in labelledUnigrams:
            if word['low'] == 0:
                word['low'] = float(1)/(len(variable_low)+1)
            else:
                word['low'] = float(word['low'])/len(variable_low)
            if word['med'] == 0:
                word['med'] = float(1)/(len(variable_med)+1)
            else:
                word['med'] = float(word['med'])/len(variable_med)
            if word['high'] == 0:
                word['high'] = float(1)/(len(variable_high)+1)
            else:
                word['high'] = float(word['high'])/len(variable_high)