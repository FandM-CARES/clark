import math
import numpy as np
np.seterr(all='raise')
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nlp_helpers import *
from graphing_helpers import *

class ClarkModel(object):

    def __init__(self):
        self.variables = ['pleasantness', 'attention', 'control',
                          'certainty', 'anticipated_effort', 'responsibility']
        self.emotions = ['sadness', 'joy', 'fear', 'anger', 'challenge', 'boredom', 'frustration']
        self.ngrams = {}
        self.decision_tree = None
        self.priors = {}
        self.variable_dimensions = ['low','med','high']
        self.micro_fscores = 0.0
        self.macro_fscores = 0.0
        self.true = []
        self.pred = []
        self.version = 2 # unigrams = 0, bigrams = 1, both = 2

    def train(self, training_data):
        '''
        Builds a trained CLARK model

        Parameters:
        training_data (array): training data used to train the model

        Returns:
        Model: a trained model
        '''
        dt = self.__build_decision_tree(training_data)

        for var in self.variables:
            ngrams, totals, var_priors, vocab = self.__train_by_variable(training_data, var)
            self.priors[var] = var_priors
            self.ngrams[var] = self.__smooth_values(ngrams, var, totals, vocab)

    def __train_by_variable(self, training_set, variable, data_points={}):
        '''
        Calculates the counts for each unigram and priors for each classification

        Parameters:
        training_set (array): training data used to train the model
        variable (string): variable in use in training

        Returns:
        Object: ngrams with associated counts
        Object: sums for each classification
        Object: priors for each classification
        '''

        words = {}
        totals = {dim:1 for dim in self.variable_dimensions}
        vocab = set()

        for row in training_set:
            for turn in ['turn1','turn2','turn3']:
                weight = self.variable_dimensions[int(row[turn]['appraisals'][variable])]
                
                tokenized_res = tokenize(row[turn]['text'])
                
                res = ngrams_and_remove_stop_words(tokenized_res, self.version)
                for word in res:
                    vocab.add(word)
                    if word in words:
                        words[word][weight] += 1
                        totals[weight] += 1
                    else:
                        words[word] = self.__initialize_av_weights()
                        words[word][weight] += 1
                        totals[weight] += 1
                
        denom = sum(totals.values())
        priors = {dim:float(totals[dim])/float(denom) for dim in self.variable_dimensions}

        return words, totals, priors, vocab

    def __initialize_av_weights(self):
        return {dim:1 for dim in self.variable_dimensions}

    def __smooth_values(self, ngrams, variable, totals, vocab):
        '''
        Performs smoothing on unigram values

        Parameters:
        ngrams (object): ngrams with associated counts in training data
        variable (string): the variable associated with the unigram values
        totals (object): total number of low, med, and high classifications for the variable

        Returns:
        Object: smoothed values for the ngrams
        '''

        len_vocab = len(vocab)

        for word in ngrams:
            for dim in self.variable_dimensions:
                ngrams[word][dim] = float(ngrams[word][dim])/float(totals[dim] + len_vocab)

        return ngrams

    def test(self, testing_data):
        '''
        Tests the precision/recall of the model

        Parameters:
        testing_data (array): data on which to test the model

        Returns:
        Null
        '''

        for row in testing_data:
            u_priors = dict(self.priors)

            tokenized_turn1 = tokenize(row['turn1']['text'])
            tokenized_turn2 = tokenize(row['turn2']['text'])
            tokenized_turn3 = tokenize(row['turn3']['text'])

            conv = tokenized_turn1 + tokenized_turn2 + tokenized_turn3

            parsed_message = flatten([ngrams_and_remove_stop_words(x, self.version) for x in [tokenized_turn1, tokenized_turn2, tokenized_turn3]])
            for var in self.variables:
                classification = self.__normalize(self.__classify(self.ngrams[var], parsed_message, u_priors[var]))
                for i,e in enumerate(self.variable_dimensions):
                    u_priors[var][e] = classification[i]

            parsed_message = ngrams_and_remove_stop_words(tokenized_turn1, self.version)
            for var in self.variables:
                classification = self.__normalize(self.__classify(self.ngrams[var], parsed_message, u_priors[var]))
                for i,e in enumerate(self.variable_dimensions):
                    u_priors[var][e] = classification[i]

            parsed_message = ngrams_and_remove_stop_words(tokenized_turn3, self.version)
            var_classification = {dim:'' for dim in self.variables}
            for var in self.variables:
                var_classification[var] = self.__classify(self.ngrams[var], parsed_message, u_priors[var], False)
            
            self.true.append(row['turn3']['emotion'])
            emo_class = self.__map_to_emotion(var_classification)
            self.pred.append(emo_class[0])

        self.__calculate_scores()

    def __classify(self, training_dict, content, priors, raw=True):
        '''
        Classifies each message according to the trained model

        Parameters:
        training_dict (Object): trained model
        content (String): message to be tested
        priors (Object): priors

        Returns:
        String: classification according to the trained model
        '''

        low = [math.log(priors['low']), 'low']
        med = [math.log(priors['med']), 'med']
        high = [math.log(priors['high']), 'high']

        for word in content:
            if word in training_dict:
                low[0] += float(math.log(training_dict[word]['low']))
                med[0] += float(math.log(training_dict[word]['med']))
                high[0] += float(math.log(training_dict[word]['high']))

        if raw: return list(map(lambda x: x[0], [low, med, high]))

        return max([low, med, high],key=lambda item:item[0])[1]

    def __normalize(self, arr):
        """
        Normalizes between 0.1 and 1.0
        """
        a = 0.9 * (arr - np.min(arr))/np.ptp(arr) + 0.1
        return a/a.sum(0)

    def __build_decision_tree(self, data):
        X = [list(d['turn3']['appraisals'].values()) for d in data]
        y = [em['turn3']['emotion'] for em in data]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=1) 
        
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

        clf = clf.fit(X_train,y_train)
        self.decision_tree = clf


    def __map_to_emotion(self, vars):
        
        v = [self.variable_dimensions.index(x) for x in list(vars.values())]
        return self.decision_tree.predict(np.asarray(v).reshape(1,-1))
        # if vars['pleasantness'] == 'high' and vars['anticipated_effort'] != 'high':
        #     return 'joy'
        # if vars['pleasantness'] == 'low':
        #     if vars['control'] == 'high':
        #         if vars['certainty'] == 'med':
        #             return 'sadness'
        #     if vars['control'] == 'low':
        #         if vars['responsibility'] == 'low':
        #             return 'anger'
        #     if vars['attention'] != 'low':
        #         return 'frustration'
        # if vars['pleasantness'] != 'high':
        #     if vars['anticipated_effort'] == 'low':
        #         if vars['attention'] == 'low':
        #             return 'boredom'
        #     if vars['anticipated_effort'] == 'high':
        #         return 'challenge'
        #     if vars['certainty'] == 'low':
        #         return 'fear'

    def __calculate_scores(self):
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
        try:
            self.micro_fscores = 2 * pi * ro / (pi + ro)
        except:
            self.micro_fscores = 0.0

        temp_macro = 0
        for e in self.emotions:
            tp_e = np.sum(np.logical_and(self.pred == e, self.true == e))
            tp_fp_e = len([x for x in self.pred if x != e])
            tp_fn_e= len([x for x in self.true if x == e])

            try:
                pi_e = tp_e / tp_fp_e
            except:
                pi_e = 0.0
            
            try:
                ro_e = tp_e / tp_fn_e
            except:
                ro_e = 0.0

            try:
                temp_macro += 2 * pi_e * ro_e / (pi_e + ro_e)
            except:
                temp_macro += 0.0
        
        self.macro_fscores = temp_macro / 7

    def confusion_matrix(self, normalize=False):
        """
        Computes the confusion matrices for each of the variables
        """

        cn_matrix = confusion_matrix(self.true, self.pred)
        plot_confusion_matrix(cn_matrix, self.emotions, 'CLARK Emotions', normalize)