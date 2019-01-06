import math
import re
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nlp_helpers import *

class Emotion_Model(object):
    
    def __init__(self):
        self.emotions = ['sadness', 'joy', 'fear', 'anger', 'challenge', 'boredom', 'frustration']
        self.unigrams = {}
        self.priors = {}
        self.micro_fscores = 0.0
        self.macro_fscores = 0.0
        self.vocab = set()
        self.true = list()
        self.pred = list()
    
    def train(self, training_data):
        """
        Builds a trained Emotions model

        Parameters:
        training_data (array): training data used to train the model

        Returns:
        Model: a trained model
        """
        words = {}
        totals = {}
        for emotion in self.emotions:
            totals[emotion] = 0

        for row in training_data:
            emotion = row['Emotion']
            res = tokenize(row['Player Message'])
            for word in res:
                self.vocab.add(word)
                if is_stop_word(word):
                    continue
                if word in words:
                    words[word][emotion] += 1
                    totals[emotion] += 1
                else:
                    words[word] = {}
                    for emotion in self.emotions:
                        words[word][emotion] = 1
                    words[word][emotion] += 1
                    totals[emotion] += 1
        
        sum_totals = sum(totals.values())
        for emotion in self.emotions:
            self.priors[emotion] = totals[emotion] / sum_totals

        self.unigrams = self.calculate_probabilities(words, totals)

    def calculate_probabilities(self, words, totals):
        for word in words:
            for emotion in self.emotions:
                words[word][emotion] = float(words[word][emotion])/float(totals[emotion]+len(self.vocab))

        return words
    
    def test(self, testing_data):
        """
        Tests the precision/recall of the model

        Parameters:
        testing_data (array): data on which to test the model

        Returns:
        Null
        """
        total = len(testing_data)
        messages = {}

        for row in testing_data:
            res = []
            emotion = row['Emotion']
            response = tokenize(row['Player Message'])
            for word in response:
                if is_stop_word(word):
                    continue
                res.append(word)
            parsed_message = ' '.join(res)
            messages[parsed_message] = {}
            self.true.append(emotion)
            classification = self.classify(self.unigrams, parsed_message, self.priors)
            self.pred.append(str(classification))

        self.calculate_scores()

    def classify(self, trainingDict, content, priors):
        """
        Classifies each message according to the trained model

        Parameters:
        trainingDict (Object): trained model
        content (String): message to be tested
        PPs (Object): priors

        Returns:
        String: classification according to the trained model
        """

        sadness = [priors['sadness'], 'sadness']
        joy = [priors['joy'], 'joy']
        fear = [priors['fear'], 'fear']
        challenge = [priors['challenge'], 'challenge']
        anger = [priors['anger'], 'anger']
        boredom = [priors['boredom'], 'boredom']
        frustration = [priors['frustration'], 'frustration']
            
        for word in content.split():
            if word in trainingDict:
                sadness[0] += float(math.log(trainingDict[word]['sadness']))
                joy[0] += float(math.log(trainingDict[word]['joy']))
                fear[0] += float(math.log(trainingDict[word]['fear']))
                challenge[0] += float(math.log(trainingDict[word]['challenge']))
                anger[0] += float(math.log(trainingDict[word]['anger']))
                boredom[0] += float(math.log(trainingDict[word]['boredom']))
                frustration[0] += float(math.log(trainingDict[word]['frustration']))

        return max([sadness, joy, fear, challenge, anger, boredom, frustration],key=lambda item:item[0])[1]

    def calculate_scores(self):
        """
        Calculates the micro and macro f scores for each emotion

        Parameters:
        None

        Returns:
        None
        """
        self.pred = np.asarray(self.pred)
        self.true = np.asarray(self.true)
            
        TP = np.sum(
            np.logical_or(
                np.logical_or(
                    np.logical_or(
                        np.logical_or(
                            np.logical_or(
                                np.logical_or(np.logical_and(self.pred == 'sadness', self.true == 'sadness'), np.logical_and(self.pred == 'joy', self.true == 'joy')),
                                np.logical_and(self.pred == 'fear', self.true == 'fear')),
                            np.logical_and(self.pred == 'anger', self.true == 'anger')),
                        np.logical_and(self.pred == 'challenge', self.true == 'challenge')),
                    np.logical_and(self.pred == 'boredom', self.true == 'boredom')),
                np.logical_and(self.pred == 'frustration', self.true == 'frustration')))
        TP_FP = len(self.pred)
        TP_FN = len(self.true)         
            
        pi = TP / TP_FP
        ro = TP / TP_FN
        self.micro_fscores = 2 * pi * ro / (pi + ro)

        temp_macro = 0
        for e in self.emotions:
            TP_e = np.sum(np.logical_and(self.pred == e, self.true == e))
            TP_FP_e = len([x for x in self.pred if x != e])
            TP_FN_e= len([x for x in self.true if x == e])

            pi_e = TP_e / TP_FP_e
            ro_e = TP_e / TP_FN_e

            if pi_e == 0: pi_e = 1
            temp_macro += 2 * pi_e * ro_e / (pi_e + ro_e)
        
        self.macro_fscores = temp_macro / 7

    def confusion_matrix(self, normalize=False):
        """
        Computes the confusion matrices for each of the variables
        """

        cn_matrix = confusion_matrix(self.true, self.pred)
        self.plot_confusion_matrix(cn_matrix, self.emotions, 'Emotions', normalize)

    def plot_confusion_matrix(self, cm, classes, title, normalize=False, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        Courtesy of scikit-learn
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrices/' + title + '.png')
        plt.close()


class CLARK_Model(object):

    def __init__(self):
        self.variables = ['Pleasantness', 'Attention', 'Control',
                          'Certainty', 'Anticipated Effort', 'Responsibililty']
        self.unigrams = {}
        self.priors = {}
        self.bounds = {}
        self.micro_fscores = {}
        self.macro_fscores = {}
        self.vocab = set()
        self.true = {}
        self.pred = {}

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

    def train_by_variable(self, training_set, variable, dataPoints={}):
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

        for row in training_set:
            dataPoints[variable].append(float(row[variable]))

        self.calc_bounds_data(dataPoints)

        for row in training_set:
            weight = self.numToClassificationWeight(row, variable)
            res = tokenize(row['Player Message'])
            for word in res:
                self.vocab.add(word)
                if is_stop_word(word):
                    continue
                if word in words:
                    words[word][weight] += 1
                    totals = self.allocateClassificationWeightToTotal(weight, totals)
                else:
                    words[word] = {'low': 1, 'med': 1, 'high': 1}
                    words[word][weight] += 1
                    totals = self.allocateClassificationWeightToTotal(weight, totals)

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
                unigrams[word]['low'])/float(totals['num_low'] + len(self.vocab))
            unigrams[word]['med'] = float(
                unigrams[word]['med'])/float(totals['num_med'] + len(self.vocab))
            unigrams[word]['high'] = float(
                unigrams[word]['high'])/float(totals['num_high'] + len(self.vocab))

        return unigrams

    def test(self, testing_data):
        """
        Tests the precision/recall of the model

        Parameters:
        testing_data (array): data on which to test the model

        Returns:
        Null
        """
        total = len(testing_data)
        messages = {}

        for var in self.variables:
            self.true[var] = []
            self.pred[var] = []

        for row in testing_data:
            res = []
            response = tokenize(row['Player Message'])
            for word in response:
                if is_stop_word(word):
                    continue
                res.append(word)
            parsed_message = ' '.join(res)
            messages[parsed_message] = {}
            for var in self.variables:
                weight = self.numToClassificationWeight(row, var)
                self.true[var].append(weight)
                messages[parsed_message][var] = weight
                classification = self.classify(self.unigrams[var], parsed_message, self.priors[var])
                self.pred[var].append(classification)

        self.calculate_scores()
    
    def calculate_scores(self):
        """
        Calculates the micro and macro f scores for each variable

        Parameters:
        None

        Returns:
        None
        """

        for var in self.variables:
            self.pred[var] = np.asarray(self.pred[var])
            self.true[var] = np.asarray(self.true[var])
            
            TP = np.sum(np.logical_or(np.logical_or(np.logical_and(self.pred[var] == 'low', self.true[var] == 'low'), np.logical_and(
                    self.pred[var] == 'med', self.true[var] == 'med')), np.logical_and(self.pred[var] == 'high', self.true[var] == 'high')))
            TP_FP = len(self.pred[var])
            TP_FN = len(self.true[var])         
            
            pi = TP / TP_FP
            ro = TP / TP_FN
            self.micro_fscores[var] = 2 * pi * ro / (pi + ro)

            temp_macro = 0
            for c in ['high', 'med', 'low']:
                TP_c = np.sum(np.logical_and(self.pred[var] == c, self.true[var] == c))
                TP_FP_c = len([x for x in self.pred[var] if x != c])
                TP_FN_c = len([x for x in self.true[var] if x == c])

                pi_c = TP_c / TP_FP_c
                ro_c = TP_c / TP_FN_c

                if pi_c == 0: pi_c = 1
                temp_macro += 2 * pi_c * ro_c / (pi_c + ro_c)
            
            self.macro_fscores[var] = temp_macro / 3


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
                sumLow += float(math.log(trainingDict[word]['low']))
                sumMed += float(math.log(trainingDict[word]['med']))
                sumHigh += float(math.log(trainingDict[word]['high']))

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

    # def plotData(self):
    #     colors = ["red", "blue", "green", "yellow", "brown", "purple"]
    #     i = 0
    #     for var in self.dataPoints:
    #         sns.distplot(self.dataPoints[var], color=colors[i], label=var)
    #         i += 1
    #     plt.legend()
    #     plt.show()

    def calc_std_data(self, dataPoints):
        """

        Calculates the standard deviation of the data points

        Parameters:
        dataPoints (object): data points for each variable

        Returns:
        Array: standard deviations for the variable
        """
        res = []
        for var in dataPoints:
            res.append([np.mean(dataPoints[var]),
                        np.std(dataPoints[var]), var])
        return res

    def calc_bounds_data(self, dataPoints):
        """
        Calculates the upper and lower bounds of the classification based on the data provided

        Parameters:
        dataPoints (Object): contains training data points for each variable

        Returns:
        None
        """
        std_data = self.calc_std_data(dataPoints)
        for var in std_data:
            lb = var[0] - (var[1] / 2)
            ub = var[0] + (var[1] / 2)
            self.bounds[var[2]] = {}
            self.bounds[var[2]]['upper'] = ub
            self.bounds[var[2]]['lower'] = lb

    def confusion_matrix(self, normalize=False):
        """
        Computes the confusion matrices for each of the variables
        """

        for var in self.variables:
            cn_matrix = confusion_matrix(self.true[var], self.pred[var])
            self.plot_confusion_matrix(
                cn_matrix, ['low', 'med', 'high'], var, normalize)

    def plot_confusion_matrix(self, cm, classes, title, normalize=False, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        Courtesy of scikit-learn
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrices/' + title + '.png')
        plt.close()
