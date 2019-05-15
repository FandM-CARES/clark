import math
import numpy as np
np.seterr(all='raise')
from sklearn.metrics import confusion_matrix
from nlp_helpers import *
from graphing_helpers import *

class ClarkModel(object):

    def __init__(self):
        self.variables = ['pleasantness', 'attention', 'control',
                          'certainty', 'anticipated_effort', 'responsibility']
        self.ngrams = {}
        self.priors = {}
        self.variable_dimensions = ['low','med','high']
        # self.bounds = {}
        self.micro_fscores = {}
        self.macro_fscores = {}
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
            ngrams, totals, var_priors, vocab = self.__train_by_variable(training_data, var)
            self.priors[var] = var_priors
            self.ngrams[var] = self.__smooth_values(ngrams, var, totals, vocab)

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
        totals = {dim:1 for dim in self.variable_dimensions}
        vocab = set()

        for row in training_set:
            for turn in ['turn1','turn2','turn3']:
                weight = self.variable_dimensions[int(row[turn][variable])]
                
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
        """
        Performs smoothing on unigram values

        Parameters:
        ngrams (object): ngrams with associated counts in training data
        variable (string): the variable associated with the unigram values
        totals (object): total number of low, med, and high classifications for the variable

        Returns:
        Object: smoothed values for the ngrams
        """

        len_vocab = len(vocab)

        for word in ngrams:
            for dim in self.variable_dimensions:
                ngrams[word][dim] = float(ngrams[word][dim])/float(totals[dim] + len_vocab)

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

            tokenized_turn1 = tokenize(row['turn1']['text'])
            tokenized_turn2 = tokenize(row['turn2']['text'])
            tokenized_turn3 = tokenize(row['turn3']['text'])

            conv = tokenized_turn1 + tokenized_turn2 + tokenized_turn3

            parsed_message = flatten([ngrams_and_remove_stop_words(x, self.version) for x in [tokenized_turn1, tokenized_turn2, tokenized_turn3]])
            for var in self.variables:
                classification = self.__classify(self.ngrams[var], parsed_message, u_priors[var])
                for i,e in enumerate(self.variable_dimensions):
                    u_priors[var][e] = classification[i]

            parsed_message = ngrams_and_remove_stop_words(tokenized_turn1, self.version)
            for var in self.variables:
                classification = self.__classify(self.ngrams[var], parsed_message, u_priors[var])
                for i,e in enumerate(self.variable_dimensions):
                    u_priors[var][e] = classification[i]

            parsed_message = ngrams_and_remove_stop_words(tokenized_turn3, self.version)
            for var in self.variables:
                weight = self.variable_dimensions[int(row['turn3'][var])]
                self.true[var].append(weight)
                classification = self.__classify(self.ngrams[var], parsed_message, u_priors[var], False)
                self.pred[var].append(classification)

        self.calculate_scores()

    def __classify(self, training_dict, content, priors, raw=True):
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
