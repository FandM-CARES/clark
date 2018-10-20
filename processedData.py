import csv
import re
import nltk

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

class ProcessedData(object):

    appraisalVariables = ['Pleasantness', 'Attention', 'Control',
                          'Certainty', 'Anticipated Effort', 'Responsibililty']

    def __init__(self, linkToRawDataSet):
        self.input = linkToRawDataSet
        self.result = self.processDataUnigrams(linkToRawDataSet)

    def processDataUnigrams(self, linkToRawDataset):
        print(self.parseCSVIntoLabelledUnigrams(linkToRawDataset))

    def parseCSVIntoLabelledUnigrams(self, linkToRawDataset, words={}):
        with open(linkToRawDataset, mode='r') as rawData:
            csv_reader = csv.DictReader(rawData)
            for row in csv_reader:
                for word in re.sub(r'[.!,;?]', " ",  row['Player Message']).split(): # substituting any non-alphanumerical with a space and then splitting on it to get words
                    if word in stop_words:
                        continue
                    word = word.lower()
                    if word in words:
                        for variable in self.appraisalVariables:
                            words[word][variable] += float(row[variable])
                    else:
                        words[word] = {}
                        for variable in self.appraisalVariables:
                            words[word][variable] = float(row[variable])
        return words, len(words)
