import math
import re
import numpy as np
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


class Model(object):

    def __init__(self):
        self.variables = ['Pleasantness', 'Attention', 'Control',
                          'Certainty', 'Anticipated Effort', 'Responsibililty']
        self.unigrams = {}
        self.priors = {}
        self.bounds = {}
        self.accuracies = {}
        self.precisions = {}
        self.recalls = {}
        self.fscores = {}

    def train(self, training_data):
        """
        Builds a trained CLARK model

        Parameters:
        training_data (array): training data used to train the model

        Returns:
        Model: a trained model
        """

        for var in self.variables:
            unigrams, totals, PPs = self.train_by_variable(training_data, var)
            self.priors[var] = PPs
            self.unigrams[var] = self.smooth_values(unigrams, var, totals)

    def train_by_variable(self, training_set, variable, dataPoints = {}):
        """
        Calculates the counts for each unigram and priors for each classification

        Parameters:
        training_set (array): training data used to train the model
        variable (string): variable in use in training

        Returns:
        Object: unigrams with associated counts
        Object: sums for each classification
        Object: priors for each classification
        """

        words = {}
        totals = {
            'num_low': 0,
            'num_med': 0,
            'num_high': 0
        }

        dataPoints[variable] = []

        # There has to be a better way to get priors
        for row in training_set:
            # weight = self.numToClassificationWeight(row, variable)
            dataPoints[variable].append(float(row[variable]))
        
        self.calcBoundsData(dataPoints)

        for row in training_set:
            weight = self.numToClassificationWeight(row, variable)
            for word in re.sub(r'[.!,;?]', " ",  row['Player Message']).split():
                if word in stop_words:
                    continue
                word = word.lower()
                if word in words:
                    words[word][weight] += 1
                    totals = self.allocateClassificationWeightToTotal(
                        weight, totals)
                else:
                    words[word] = {'low': 1, 'med': 1, 'high': 1}
                    words[word][weight] += 1
                    totals = self.allocateClassificationWeightToTotal(
                        weight, totals)

        PPs = {
            'low': float(totals['num_low'])/float(totals['num_low'] + totals['num_med'] + totals['num_high']),
            'med': float(totals['num_med'])/float(totals['num_low'] + totals['num_med'] + totals['num_high']),
            'high': float(totals['num_high'])/float(totals['num_low'] + totals['num_med'] + totals['num_high'])
        }

        return words, totals, PPs

    def allocateClassificationWeightToTotal(self, cw, totals):
        """
        Updates the totals

        Parameters:
        cw (string): classification weight
        totals (object): totals to be updates

        Returns:
        Object: totals
        """

        if cw == 'low':
            totals['num_low'] += 1
        elif cw == 'med':
            totals['num_med'] += 1
        elif cw == 'high':
            totals['num_high'] += 1
        return totals

    def numToClassificationWeight(self, row, variable):
        """
        Converts number values to a classification weighting

        Parameters:
        row (array): row of data contianing variables
        variable (string): the variable to look at

        Returns:
        String: classification weight

        TODO: update bounds automatically
        """

        def numToCWForPleasantness(row):
            if float(row['Pleasantness']) <= self.bounds['Pleasantness']['lower']:
                return 'low'
            if float(row['Pleasantness']) <= self.bounds['Pleasantness']['upper']:
                return 'med'
            return 'high'

        def numToCWForAttention(row):
            if float(row['Attention']) <= self.bounds['Attention']['lower']:
                return 'low'
            if float(row['Attention']) <= self.bounds['Attention']['upper']:
                return 'med'
            return 'high'

        def numToCWForControl(row):
            if float(row['Control']) <= self.bounds['Control']['lower']:
                return 'low'
            if float(row['Control']) <= self.bounds['Control']['upper']:
                return 'med'
            return 'high'

        def numToCWForCertainty(row):
            if float(row['Certainty']) <= self.bounds['Certainty']['lower']:
                return 'low'
            if float(row['Certainty']) <= self.bounds['Certainty']['upper']:
                return 'med'
            return 'high'

        def numToCWForAnticipatedEffort(row):
            if float(row['Anticipated Effort']) <= self.bounds['Anticipated Effort']['lower']:
                return 'low'
            if float(row['Anticipated Effort']) <= self.bounds['Anticipated Effort']['upper']:
                return 'med'
            return 'high'

        def numToCWForResponsibility(row):
            if float(row['Responsibililty']) <= self.bounds['Responsibililty']['lower']:
                return 'low'
            if float(row['Responsibililty']) <= self.bounds['Responsibililty']['upper']:
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

    def smooth_values(self, unigrams, variable, totals):
        """
        Performs smoothing on unigram values

        Parameters:
        unigrams (object): unigrams with associated counts in training data
        variable (string): the variable associated with the unigram values
        totals (object): total number of low, med, and high classifications for the variable

        Returns:
        Object: smoothed values for the unigrams
        """

        for word in unigrams:
            unigrams[word]['low'] = float(
                unigrams[word]['low'])/float(totals['num_low'])
            unigrams[word]['med'] = float(
                unigrams[word]['med'])/float(totals['num_med'])
            unigrams[word]['high'] = float(
                unigrams[word]['high'])/float(totals['num_high'])

        return unigrams

    def test(self, testing_data):
        """
        Tests the accuracy of the model

        Parameters:
        testing_data (array): data on which to test the model

        Returns:
        Null
        """
        correct = 0
        total = len(testing_data)
        messages = {}

        TP = {}
        TP_FP = {}
        TP_FN = {}

        for var in self.variables:
            self.accuracies[var] = 0
            TP[var] = {
                'high': 0,
                'med': 0,
                'low': 0
            }
            TP_FP[var] = {
                'high': 0,
                'med': 0,
                'low': 0
            }
            TP_FN[var] = {
                'high': 0,
                'med': 0,
                'low': 0
            }

        for row in testing_data:
            res = []
            for word in re.sub(r'[.!,;?]', " ",  row['Player Message']).split():
                if word in stop_words:
                    continue
                word = word.lower()
                res.append(word)
            parsed_message = ' '.join(res)
            messages[parsed_message] = {}
            for var in self.variables:
                weight = self.numToClassificationWeight(row, var)
                TP_FN[var][weight] += 1
                messages[parsed_message][var] = weight
                classification = self.classify(
                    self.unigrams[var], parsed_message, self.priors[var])
                TP_FP[var][classification] += 1
                if classification == messages[parsed_message][var]:
                    self.accuracies[var] += 1
                    TP[var][classification] += 1

        for var in self.accuracies:
            self.accuracies[var] = float(self.accuracies[var]/total)
            mean_precision = 0
            mean_recall = 0
            for label in ['high', 'med', 'low']:
                mean_precision += float(TP[var][label]/TP_FP[var][label])
                mean_recall += float(TP[var][label]/TP_FN[var][label])
            self.precisions[var] = float(mean_precision/3)
            self.recalls[var] = float(mean_recall/3)

            self.fscores[var] = (2 * self.precisions[var] * self.recalls[var]
                                 )/(self.precisions[var] + self.recalls[var])

    def classify(self, trainingDict, content, PPs):
        """
        Classifies each message according to the trained model

        Parameters:
        trainingDict (Object): trained model
        content (String): message to be tested
        PPs (Object): priors

        Returns:
        String: classification according to the trained model
        """
        sumLow = float(0)
        sumMed = float(0)
        sumHigh = float(0)

        for word in content.split():
            if word in trainingDict:
                sumLow += float(math.log(trainingDict[word]['low'], 10))
                sumMed += float(math.log(trainingDict[word]['med'], 10))
                sumHigh += float(math.log(trainingDict[word]['high'], 10))

        lowProb = math.log(PPs['low'], 10) + sumLow
        medProb = math.log(PPs['med'], 10) + sumMed
        highProb = math.log(PPs['high'], 10) + sumHigh

        maxVal = max([lowProb, medProb, highProb])
        if lowProb == maxVal:
            return 'low'
        if medProb == maxVal:
            return 'med'
        else:
            return 'high'

    # def plotData(self):
    #     colors = ["red", "blue", "green", "yellow", "brown", "purple"]
    #     i = 0
    #     for var in self.dataPoints:
    #         sns.distplot(self.dataPoints[var], color=colors[i], label=var)
    #         i += 1
    #     plt.legend()
    #     plt.show()

    def calcSTDData(self, dataPoints):
        res = []
        for var in dataPoints:
            res.append([np.mean(dataPoints[var]), np.std(dataPoints[var]), var])
        return res

    def calcBoundsData(self, dataPoints):
        stdData = self.calcSTDData(dataPoints)
        for var in stdData:
            lb = var[0] - (var[1] / 2)
            ub = var[0] + (var[1] / 2)
            self.bounds[var[2]] = {}
            self.bounds[var[2]]['upper'] = ub
            self.bounds[var[2]]['lower'] = lb
