import math
import numpy as np
np.seterr(all='raise')
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from nlp_helpers import *
from graphing_helpers import *

class ClarkModel(object):

    def __init__(self):
        self.variables = ['pleasantness', 'attention', 'control',
                          'certainty', 'anticipated_effort', 'responsibility']
        self.emotions = ['sadness', 'joy', 'fear', 'anger', 'challenge', 'boredom', 'frustration']
        self.ngrams = {}
        self.av2e_model = None
        self.priors = {}
        self.variable_dimensions = ['low','med','high']
        self.micro_fscores = 0.0
        self.macro_fscores = 0.0
        self.tense = {var:self.__init_tense() for var in self.variables}
        self.pronouns = {var:self.__init_pronouns() for var in self.variables}
        self.true = []
        self.pred = []
        self.version = 2 # unigrams = 0, bigrams = 1, both = 2
    
    def __init_tense(self):
        return {
            'past': {dim: 1 for dim in self.variable_dimensions},
            'present': {dim: 1 for dim in self.variable_dimensions},
            'future': {dim: 1 for dim in self.variable_dimensions}
        }
    
    def __init_pronouns(self):
        return {
            'first': {dim: 1 for dim in self.variable_dimensions},
            'second': {dim: 1 for dim in self.variable_dimensions},
            'third': {dim: 1 for dim in self.variable_dimensions}
        }

    def train(self, training_data):
        """
        Builds a trained CLARK model

        Parameters:
        training_data (array): training data used to train the model

        Returns:
        Model: a trained model
        """
        self.__build_av2e_model(training_data)

        for var in self.variables:
            self.__train_by_variable(training_data, var)
            
    def __train_by_variable(self, training_set, variable, data_points={}):
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
        words_totals = {dim:0 for dim in self.variable_dimensions}
        tense_totals = {dim:0 for dim in self.variable_dimensions}
        pronoun_totals = {dim:0 for dim in self.variable_dimensions}
        words_vocab = set()
        tense_vocab = set()
        pronoun_vocab = set()

        for row in training_set:
            for turn in ['turn1','turn2','turn3']:
                true_dim = self.variable_dimensions[int(row[turn]['appraisals'][variable])]                
                tokenized_res = tokenize(row[turn]['text'])
                
                pos = parts_of_speech(tokenized_res)
                for p in pos:
                    p_tense = determine_tense(p)
                    p_pronoun = determine_pronoun(p)
                    if p_tense != "": 
                        tense_vocab.add(p_tense)
                        self.tense[variable][p_tense][true_dim] += 1
                        tense_totals[true_dim] += 1
                    if p_pronoun != "":
                        pronoun_vocab.add(p_pronoun)
                        self.pronouns[variable][p_pronoun][true_dim] += 1
                        pronoun_totals[true_dim] += 1

                res = ngrams_and_remove_stop_words(tokenized_res, self.version)
                for word in res:
                    words_vocab.add(word)
                    if word in words:
                        words[word][true_dim] += 1
                        words_totals[true_dim] += 1
                    else:
                        words[word] = self.__initialize_av_weights()
                        words[word][true_dim] += 1
                        words_totals[true_dim] += 1
                
        denom = sum(words_totals.values())
        self.priors[variable] = {dim:float(words_totals[dim])/float(denom) for dim in self.variable_dimensions}
        
        self.__calculate_probabilities(words, words_totals, words_vocab, tense_totals, tense_vocab, pronoun_totals, pronoun_vocab, variable)


    def __initialize_av_weights(self):
        return {dim:1 for dim in self.variable_dimensions}

    def __calculate_probabilities(self, words, words_totals, words_vocab, tense_totals, tense_vocab, pronoun_totals, pronoun_vocab, curr_var):
        """
        TODO
        """

        len_vocab = len(words_vocab)

        for word in words:
            for dim in self.variable_dimensions:
                words[word][dim] = float(words[word][dim])/float(words_totals[dim] + len_vocab)

        self.ngrams[curr_var] = words

        for tense in ['past','present','future']:
            for dim in self.variable_dimensions:
                self.tense[curr_var][tense][dim] = float(self.tense[curr_var][tense][dim])/float(tense_totals[dim]+len(tense_vocab))

        for pronoun in ['first','second','third']:
            for dim in self.variable_dimensions:
                self.pronouns[curr_var][pronoun][dim] = float(self.pronouns[curr_var][pronoun][dim])/float(pronoun_totals[dim]+len(pronoun_vocab))

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
                classification = self.__normalize(self.__classify(self.ngrams[var], parsed_message, conv, u_priors[var], var))
                for i,e in enumerate(self.variable_dimensions):
                    u_priors[var][e] = classification[i]

            parsed_message = ngrams_and_remove_stop_words(tokenized_turn1, self.version)
            for var in self.variables:
                classification = self.__normalize(self.__classify(self.ngrams[var], parsed_message, tokenized_turn1, u_priors[var], var))
                for i,e in enumerate(self.variable_dimensions):
                    u_priors[var][e] = classification[i]

            parsed_message = ngrams_and_remove_stop_words(tokenized_turn3, self.version)
            var_classification = {dim:'' for dim in self.variables}
            for var in self.variables:
                var_classification[var] = self.__classify(self.ngrams[var], parsed_message, tokenized_turn3, u_priors[var], var, False)
            
            self.true.append(row['turn3']['emotion'])
            emo_class = self.__map_to_emotion(var_classification)
            self.pred.append(emo_class[0])

        self.__calculate_scores()
    
    def __classify(self, training_dict, content, tokenized_content, priors, curr_var, raw=True):
        """
        Classifies each message according to the trained model

        Parameters:
        training_dict (Object): trained model
        content (String): message to be tested
        priors (Object): priors

        Returns:
        String: classification according to the trained model
        """

        low = [math.log(priors['low']), 'low']
        med = [math.log(priors['med']), 'med']
        high = [math.log(priors['high']), 'high']

        pos = parts_of_speech(tokenized_content)
        for p in pos:
            tense = determine_tense(p)
            pronoun = determine_pronoun(p)

            if tense in self.tense[curr_var]:
                low[0] += float(math.log(self.tense[curr_var][tense]['low']))
                med[0] += float(math.log(self.tense[curr_var][tense]['med']))
                high[0] += float(math.log(self.tense[curr_var][tense]['high']))

            if tense in self.pronouns[curr_var]:
                low[0] += float(math.log(self.pronouns[curr_var][pronoun]['low']))
                med[0] += float(math.log(self.pronouns[curr_var][pronoun]['med']))
                high[0] += float(math.log(self.pronouns[curr_var][pronoun]['high']))

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

    def __build_av2e_model(self, data):
        X = [list(d['turn3']['appraisals'].values()) for d in data]
        y = [em['turn3']['emotion'] for em in data]      

        rf = RandomForestClassifier(n_estimators=10)
        rf = rf.fit(X,y)

        self.av2e_model = rf

    def __map_to_emotion(self, vars):
        v = [self.variable_dimensions.index(x) for x in list(vars.values())]
        return self.av2e_model.predict(np.asarray(v).reshape(1,-1))

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