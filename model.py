import math
import re
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nlp_helpers import *
from graphing_helpers import *

class EmotionModel(object):
    
    def __init__(self):
        self.emotions = ['sadness', 'joy', 'fear', 'anger', 'challenge', 'boredom', 'frustration']
        self.ngrams = {}
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

        self.ngrams = self.calculate_probabilities(words, totals)

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
        messages = {}

        for row in testing_data:
            res = []
            emotion = row['Emotion']
            response = tokenize(row['Player Message'])
            for word in response:
                res.append(word)
            parsed_message = ' '.join(res)
            messages[parsed_message] = {}
            self.true.append(emotion)
            classification = self.classify(self.ngrams, parsed_message, self.priors)
            self.pred.append(str(classification))

        self.calculate_scores()

    def classify(self, training_dict, content, priors):
        """
        Classifies each message according to the trained model

        Parameters:
        training_dict (Object): trained model
        content (String): message to be tested
        priors (Object): priors

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
            if word in training_dict:
                sadness[0] += float(math.log(training_dict[word]['sadness']))
                joy[0] += float(math.log(training_dict[word]['joy']))
                fear[0] += float(math.log(training_dict[word]['fear']))
                challenge[0] += float(math.log(training_dict[word]['challenge']))
                anger[0] += float(math.log(training_dict[word]['anger']))
                boredom[0] += float(math.log(training_dict[word]['boredom']))
                frustration[0] += float(math.log(training_dict[word]['frustration']))

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
            
        tp = np.sum(
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
        tp_fp = len(self.pred)
        tp_fn = len(self.true)         
            
        pi = tp / tp_fp
        ro = tp / tp_fn
        self.micro_fscores = 2 * pi * ro / (pi + ro)

        temp_macro = 0
        for e in self.emotions:
            tp_e = np.sum(np.logical_and(self.pred == e, self.true == e))
            tp_fp_e = len([x for x in self.pred if x != e])
            tp_fn_e= len([x for x in self.true if x == e])

            pi_e = tp_e / tp_fp_e
            ro_e = tp_e / tp_fn_e

            if pi_e == 0: pi_e = 1
            temp_macro += 2 * pi_e * ro_e / (pi_e + ro_e)
        
        self.macro_fscores = temp_macro / 7

    def confusion_matrix(self, normalize=False):
        """
        Computes the confusion matrices for each of the variables
        """

        cn_matrix = confusion_matrix(self.true, self.pred)
        plot_confusion_matrix(cn_matrix, self.emotions, 'Emotions', normalize)

class ClarkModel(object):

    def __init__(self):
        self.variables = ['Pleasantness', 'Attention', 'Control',
                          'Certainty', 'Anticipated Effort', 'Responsibililty']
        self.ngrams = {}
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
            ngrams, totals, priors = self.train_by_variable(training_data, var)
            self.priors[var] = priors
            self.ngrams[var] = self.smooth_values(ngrams, var, totals)

    def train_by_variable(self, training_set, variable, data_points={}):
        """
        Calculates the counts for each unigram and priors for each classification

        Parameters:
        training_set (array): training data used to train the model
        variable (string): variable in use in training

        Returns:
        Object: ngrams with associated counts
        Object: sums for each classification
        Object: priors for each classification
        """

        words = {}
        totals = {
            'num_low': 0,
            'num_med': 0,
            'num_high': 0
        }

        data_points[variable] = []

        for row in training_set:
            data_points[variable].append(float(row[variable]))

        self.calc_bounds_data(data_points)

        for row in training_set:
            weight = self.num_to_weight(row, variable)
            res = tokenize(row['Player Message'])
            for i, word in enumerate(res):
                temp_bigram = word + res[i+1] if i + 1 < len(res) else ""
                self.vocab.add(word)
                if temp_bigram != "": self.vocab.add(temp_bigram) 

                if word in words:
                    words[word][weight] += 1
                    totals = self.add_weight_to_total(weight, totals)
                else:
                    words[word] = self.initialize_av_weights()
                    words[word][weight] += 1
                    totals = self.add_weight_to_total(weight, totals)
                
                if temp_bigram in words:
                    words[temp_bigram][weight] += 1
                    totals = self.add_weight_to_total(weight, totals)
                elif temp_bigram != "":
                    words[temp_bigram] = self.initialize_av_weights()
                    words[temp_bigram][weight] += 1
                    totals = self.add_weight_to_total(weight, totals)
                   
                    
        priors = {
            'low': float(totals['num_low'])/float(totals['num_low'] + totals['num_med'] + totals['num_high']),
            'med': float(totals['num_med'])/float(totals['num_low'] + totals['num_med'] + totals['num_high']),
            'high': float(totals['num_high'])/float(totals['num_low'] + totals['num_med'] + totals['num_high'])
        }

        return words, totals, priors

    def initialize_av_weights(self):
        return {'low': 1, 'med': 1, 'high': 1}

    def add_weight_to_total(self, cw, totals):
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

    def num_to_weight(self, row, variable):
        """
        Converts number values to a classification weighting

        Parameters:
        row (array): row of data contianing variables
        variable (string): the variable to look at

        Returns:
        String: classification weight
        """

        def num_to_weight_pleasantness(row):
            if float(row['Pleasantness']) <= self.bounds['Pleasantness']['lower']:
                return 'low'
            if float(row['Pleasantness']) <= self.bounds['Pleasantness']['upper']:
                return 'med'
            return 'high'

        def num_to_weight_attention(row):
            if float(row['Attention']) <= self.bounds['Attention']['lower']:
                return 'low'
            if float(row['Attention']) <= self.bounds['Attention']['upper']:
                return 'med'
            return 'high'

        def num_to_weight_control(row):
            if float(row['Control']) <= self.bounds['Control']['lower']:
                return 'low'
            if float(row['Control']) <= self.bounds['Control']['upper']:
                return 'med'
            return 'high'

        def num_to_weight_certainty(row):
            if float(row['Certainty']) <= self.bounds['Certainty']['lower']:
                return 'low'
            if float(row['Certainty']) <= self.bounds['Certainty']['upper']:
                return 'med'
            return 'high'

        def num_to_weight_anticipated_effort(row):
            if float(row['Anticipated Effort']) <= self.bounds['Anticipated Effort']['lower']:
                return 'low'
            if float(row['Anticipated Effort']) <= self.bounds['Anticipated Effort']['upper']:
                return 'med'
            return 'high'

        def num_to_weight_responsibility(row):
            if float(row['Responsibililty']) <= self.bounds['Responsibililty']['lower']:
                return 'low'
            if float(row['Responsibililty']) <= self.bounds['Responsibililty']['upper']:
                return 'med'
            return 'high'

        if variable == 'Pleasantness':
            return num_to_weight_pleasantness(row)
        if variable == 'Attention':
            return num_to_weight_attention(row)
        if variable == 'Control':
            return num_to_weight_control(row)
        if variable == 'Certainty':
            return num_to_weight_certainty(row)
        if variable == 'Anticipated Effort':
            return num_to_weight_anticipated_effort(row)
        if variable == 'Responsibililty':
            return num_to_weight_responsibility(row)

    def smooth_values(self, ngrams, variable, totals):
        """
        Performs smoothing on unigram values

        Parameters:
        ngrams (object): ngrams with associated counts in training data
        variable (string): the variable associated with the unigram values
        totals (object): total number of low, med, and high classifications for the variable

        Returns:
        Object: smoothed values for the ngrams
        """

        for word in ngrams:
            ngrams[word]['low'] = float(
                ngrams[word]['low'])/float(totals['num_low'] + len(self.vocab))
            ngrams[word]['med'] = float(
                ngrams[word]['med'])/float(totals['num_med'] + len(self.vocab))
            ngrams[word]['high'] = float(
                ngrams[word]['high'])/float(totals['num_high'] + len(self.vocab))

        return ngrams

    def test(self, testing_data):
        """
        Tests the precision/recall of the model

        Parameters:
        testing_data (array): data on which to test the model

        Returns:
        Null
        """
        messages = {}

        for var in self.variables:
            self.true[var] = []
            self.pred[var] = []

        for row in testing_data:
            res = []
            response = tokenize(row['Player Message'])
            for word in response:
                res.append(word)
            parsed_message = ' '.join(res)
            messages[parsed_message] = {}
            for var in self.variables:
                weight = self.num_to_weight(row, var)
                self.true[var].append(weight)
                messages[parsed_message][var] = weight
                classification = self.classify(self.ngrams[var], parsed_message, self.priors[var])
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
            
            tp = np.sum(np.logical_or(np.logical_or(np.logical_and(self.pred[var] == 'low', self.true[var] == 'low'), np.logical_and(
                    self.pred[var] == 'med', self.true[var] == 'med')), np.logical_and(self.pred[var] == 'high', self.true[var] == 'high')))
            tp_fp = len(self.pred[var])
            tp_fn = len(self.true[var])         
            
            pi = tp / tp_fp
            ro = tp / tp_fn
            self.micro_fscores[var] = 2 * pi * ro / (pi + ro)

            temp_macro = 0
            for c in ['high', 'med', 'low']:
                tp_c = np.sum(np.logical_and(self.pred[var] == c, self.true[var] == c))
                tp_fp_c = len([x for x in self.pred[var] if x != c])
                tp_fn_c = len([x for x in self.true[var] if x == c])

                pi_c = tp_c / tp_fp_c
                ro_c = tp_c / tp_fn_c

                if pi_c == 0: pi_c = 1
                temp_macro += 2 * pi_c * ro_c / (pi_c + ro_c)
            
            self.macro_fscores[var] = temp_macro / 3


    def classify(self, training_dict, content, priors):
        """
        Classifies each message according to the trained model

        Parameters:
        training_dict (Object): trained model
        content (String): message to be tested
        priors (Object): priors

        Returns:
        String: classification according to the trained model
        """
        sum_low = float(0)
        sum_med = float(0)
        sum_high = float(0)

        res = content.split()
        for i, word in enumerate(res):
            if word in training_dict:
                sum_low += float(math.log(training_dict[word]['low']))
                sum_med += float(math.log(training_dict[word]['med']))
                sum_high += float(math.log(training_dict[word]['high']))
            temp_bigram = word + res[i + 1] if i + 1 < len(res) else ''
            if temp_bigram in training_dict and temp_bigram != '':
                sum_low += float(math.log(training_dict[temp_bigram]['low']))
                sum_med += float(math.log(training_dict[temp_bigram]['med']))
                sum_high += float(math.log(training_dict[temp_bigram]['high']))

        low_prob = math.log(priors['low']) + sum_low
        med_prob = math.log(priors['med']) + sum_med
        high_prob = math.log(priors['high']) + sum_high

        max_val = max([low_prob, med_prob, high_prob])
        if low_prob == max_val:
            return 'low'
        if med_prob == max_val:
            return 'med'
        else:
            return 'high'

    def calc_std_data(self, data_points):
        """

        Calculates the standard deviation of the data points

        Parameters:
        data_points (object): data points for each variable

        Returns:
        Array: standard deviations for the variable
        """
        res = []
        for var in data_points:
            res.append([np.mean(data_points[var]),
                        np.std(data_points[var]), var])
        return res

    def calc_bounds_data(self, data_points):
        """
        Calculates the upper and lower bounds of the classification based on the data provided

        Parameters:
        data_points (Object): contains training data points for each variable

        Returns:
        None
        """
        std_data = self.calc_std_data(data_points)
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
            plot_confusion_matrix(cn_matrix, ['low', 'med', 'high'], var, normalize)
