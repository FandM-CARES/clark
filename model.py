from processedData import *

class Model:

    def __init__(self, datasets=[]):
        self.datasets = datasets
        self.trained = False
        self.accuracy = 0
    
    def processData(self, datasets):
        for dset in datasets:
            test = np.spli
            pdata = ProcessedData(dset)
            normalizedDict = self.normalizeValues(pdata.unigrams, pdata.totals)

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

    def classify(self, variable, trainingDict, content, totals, PPs):
        sumLow = float(0)
        sumMed = float(0)
        sumHigh = float(0)

        for word in content:
            if x in trainingDict:
                sumLow += float(math.log(trainingDict[variable]['low'],10))
                sumMed += float(math.log(trainingDict[variable]['med'], 10))
                sumHigh += float(math.log(trainingDict[variable]['high'], 10))
        
        lowProb = math.log(PPs['low']) + sumLow
        medProb = math.log(PPs['med']) + sumMed
        highProb = math.log(PPs['high']) + sumHigh

        maxVal =  max([lowProb, medProb, highProb])
        if lowProb == maxVal:
            return 'low'
        if medProb == maxVal:
            return 'med'
        else:
            return 'high'

    def sort(self, processedData):
        print('hi')


    def nfold(X, Y, n):
        x_set = np.split(X, n)
        y_set = np.split(Y, n)
        return x_set, y_set        