import math
import re
import numpy as np
np.seterr(all='raise')
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
        self.true = list()
        self.pred = list()
        self.tense = self.__init_tense()
        # self.pronouns = {}
        self.version = 2 # 0 - unigrams, 1 - bigrams, 2, both
    
    def __init_tense(self):
        return {
            'past': {emotion: 1 for emotion in self.emotions},
            'present': {emotion: 1 for emotion in self.emotions},
            'future': {emotion: 1 for emotion in self.emotions}
        }

    def train(self, training_data):
        """
        Builds a trained Emotions model

        Parameters:
        training_data (array): training data used to train the model

        Returns:
        Model: a trained model
        """
        words = {}
        words_vocab = set()
        tense_vocab = set()
        words_totals = {emotion:0 for emotion in self.emotions}
        tense_totals = {emotion:0 for emotion in self.emotions}

        for row in training_data:
            for turn in ['turn1', 'turn2', 'turn3']:
                emotion = row[turn]['emotion']
                tokenized_res = tokenize(row[turn]['text'])
                
                pos = parts_of_speech(tokenized_res)
                for p in pos:
                    p_tense = determine_tense(p)
                    if p_tense != "": 
                        tense_vocab.add(p_tense[0])
                        self.tense[p_tense][emotion] += 1
                        tense_totals[emotion] += 1
    
                res = ngrams_and_remove_stop_words(tokenized_res, self.version)
                
                for word in res:
                    words_vocab.add(word)
                    if word in words:
                        words[word][emotion] += 1
                        words_totals[emotion] += 1
                    else:
                        words[word] = {emotion:1 for emotion in self.emotions}
                        words[word][emotion] += 1
                        words_totals[emotion] += 1
        
        sum_totals = sum(words_totals.values())
        for emotion in self.emotions:
            self.priors[emotion] = words_totals[emotion] / sum_totals

        self.__calculate_probabilities(words, words_totals, words_vocab, tense_totals, tense_vocab)

    def __calculate_probabilities(self, words, words_totals, words_vocab, tense_totals, tense_vocab):
        """
        TODO
        """

        for word in words:
            for emotion in self.emotions:
                words[word][emotion] = float(words[word][emotion])/float(words_totals[emotion]+len(words_vocab))
        
        self.ngrams = words

        for tense in self.tense:
            for emotion in self.emotions:
                self.tense[tense][emotion] = float(self.tense[tense][emotion])/float(tense_totals[emotion]+len(tense_vocab))

    def test(self, testing_data):
        """
        Tests the precision/recall of the model

        Parameters:
        testing_data (array): data on which to test the model

        Returns:
        Null
        """

        for row in testing_data:
            u_priors = dict(self.priors)

            tokenized_turn1 = tokenize(row['turn1']['text'])
            tokenized_turn2 = tokenize(row['turn2']['text'])
            tokenized_turn3 = tokenize(row['turn3']['text'])

            conv = tokenized_turn1 + tokenized_turn2 + tokenized_turn3

            parsed_message = flatten([ngrams_and_remove_stop_words(x, self.version) for x in [tokenized_turn1, tokenized_turn2, tokenized_turn3]])
            classification = self.__classify(self.ngrams, parsed_message, conv, u_priors)
            for i, e in enumerate(self.emotions):
                u_priors[e] = classification[i]

            parsed_message = ngrams_and_remove_stop_words(tokenized_turn1, self.version)
            classification = self.__classify(self.ngrams, parsed_message, tokenized_turn1, u_priors)
            for i, e in enumerate(self.emotions):
                u_priors[e] = classification[i]
            
            emotion = row['turn3']['emotion']
            self.true.append(emotion)

            parsed_message = ngrams_and_remove_stop_words(tokenized_turn3, self.version)
            classification = self.__classify(self.ngrams, parsed_message, tokenized_turn3, u_priors, False)
            
            self.pred.append(str(classification))

        self.__calculate_scores()

    def __classify(self, training_dict, content, tokenized_content, priors, raw=True):
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
        
        pos = parts_of_speech(tokenized_content)
        for p in pos:
            tense = determine_tense(p)
            if tense in self.tense:
                sadness[0] += float(math.log(self.tense[tense]['sadness']))
                joy[0] += float(math.log(self.tense[tense]['joy']))
                fear[0] += float(math.log(self.tense[tense]['fear']))
                challenge[0] += float(math.log(self.tense[tense]['challenge']))
                anger[0] += float(math.log(self.tense[tense]['anger']))
                boredom[0] += float(math.log(self.tense[tense]['boredom']))
                frustration[0] += float(math.log(self.tense[tense]['frustration']))

        for word in content:
            if word in training_dict:
                sadness[0] += float(math.log(training_dict[word]['sadness']))
                joy[0] += float(math.log(training_dict[word]['joy']))
                fear[0] += float(math.log(training_dict[word]['fear']))
                challenge[0] += float(math.log(training_dict[word]['challenge']))
                anger[0] += float(math.log(training_dict[word]['anger']))
                boredom[0] += float(math.log(training_dict[word]['boredom']))
                frustration[0] += float(math.log(training_dict[word]['frustration']))

        if raw: return list(map(lambda x: x[0], [sadness, joy, fear, challenge, anger, boredom, frustration]))

        return max([sadness, joy, fear, challenge, anger, boredom, frustration],key=lambda item:item[0])[1]

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
        plot_confusion_matrix(cn_matrix, self.emotions, 'Emotions', normalize)

class ClarkModel(object):

    def __init__(self):
        self.variables = ['pleasantness', 'attention', 'control',
                          'certainty', 'anticipated_effort', 'responsibility']
        self.ngrams = {}
        self.priors = {}
        self.variable_dimensions = ['low','med','high']
        self.bounds = {}
        self.micro_fscores = {}
        self.macro_fscores = {}
        self.vocab = set()
        self.true = {}
        self.pred = {}
        self.version = 2 # unigrams = 0, bigrams = 1, both = 2

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
            for turn in ['turn1', 'turn2', 'turn3']:
                data_points[variable].append(float(row[turn][variable]))

        for row in training_set:
            for turn in ['turn1','turn2','turn3']:
                weight = self.variable_dimensions[int(row[turn][variable])]
                parsed_message = tokenize(row[turn]['text'], self.version)
                for i, word in enumerate(parsed_message):
                    self.vocab.add(word)
                    if word in words:
                        words[word][weight] += 1
                        totals = self.add_weight_to_total(weight, totals)
                    else:
                        words[word] = self.initialize_av_weights()
                        words[word][weight] += 1
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

        for var in self.variables:
            self.true[var] = []
            self.pred[var] = []

        for row in testing_data:
            u_priors = dict(self.priors)

            parsed_message = flatten([tokenize(row[turn]['text']) for turn in ['turn1', 'turn2', 'turn3']])
            for var in self.variables:
                classification = self.classify(self.ngrams[var], parsed_message, u_priors[var])
                for i,e in enumerate(self.variable_dimensions):
                    u_priors[var][e] = classification[i]

            parsed_message = tokenize(row['turn1']['text'])
            for var in self.variables:
                classification = self.classify(self.ngrams[var], parsed_message, u_priors[var])
                for i,e in enumerate(self.variable_dimensions):
                    u_priors[var][e] = classification[i]

            parsed_message = tokenize(row['turn3']['text'])
            for var in self.variables:
                weight = self.variable_dimensions[int(row['turn3'][var])]
                self.true[var].append(weight)
                classification = self.classify(self.ngrams[var], parsed_message, u_priors[var], False)
                self.pred[var].append(classification)

        self.calculate_scores()

    def classify(self, training_dict, content, priors, raw=True):
        """
        Classifies each message according to the trained model

        Parameters:
        training_dict (Object): trained model
        content (String): message to be tested
        priors (Object): priors

        Returns:
        String: classification according to the trained model
        """

        low = [priors['low'], 'low']
        med = [priors['med'], 'med']
        high = [priors['high'], 'high']

        for word in content:
            if word in training_dict:
                low[0] += float(math.log(training_dict[word]['low']))
                med[0] += float(math.log(training_dict[word]['med']))
                high[0] += float(math.log(training_dict[word]['high']))

        if raw: return list(map(lambda x: x[0], [low, med, high]))

        return max([low, med, high],key=lambda item:item[0])[1]
    
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

            try:
                self.micro_fscores[var] = 2 * pi * ro / (pi + ro)
            except:
                self.micro_fscores[var] = 0.0

            temp_macro = 0
            for c in ['high', 'med', 'low']:
                tp_c = np.sum(np.logical_and(self.pred[var] == c, self.true[var] == c))
                tp_fp_c = len([x for x in self.pred[var] if x != c])
                tp_fn_c = len([x for x in self.true[var] if x == c])

                try:
                    pi_c = tp_c / tp_fp_c
                except:
                    pi_c = 0.0
                
                try:
                    ro_c = tp_c / tp_fn_c
                except:
                    ro_c = 0.0

                try:
                    temp_macro += 2 * pi_c * ro_c / (pi_c + ro_c)
                except:
                    temp_macro += 0.0
                
            
            self.macro_fscores[var] = temp_macro / 3
    
    def confusion_matrix(self, normalize=False):
        """
        Computes the confusion matrices for each of the variables
        """

        for var in self.variables:
            cn_matrix = confusion_matrix(self.true[var], self.pred[var])
            plot_confusion_matrix(cn_matrix, ['low', 'med', 'high'], var, normalize)
