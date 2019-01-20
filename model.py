import math
import re
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nlp_helpers import *
from graphing_helpers import *
from baseline import *

class EmotionModel(object):
    
    def __init__(self):
        self.emotions = ['happy', 'sad', 'angry', 'others']
        self.ngrams = {}
        self.priors = {}
        self.vocab = set()
        self.true = list()
        self.pred = list()
        self.emotion2encoding = {
            'others': [1,0,0,0],
            'happy': [0,1,0,0],
            'sad': [0,0,1,0],
            'angry': [0,0,0,1]
        }
        self.version = 2 # 0 - unigrams, 1 - bigrams, 2, both
        self.bag_of_words = 1 # 0 is off, 1 is on
        self.weighting_method = 1 # 0 is off, 1 is on
    
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
            emotion = row['label']
            if self.bag_of_words == 1:
                res = tokenize(row['turn1'], self.version)[0] + tokenize(row['turn2'], self.version)[0] + tokenize(row['turn3'], self.version)[0]
            else:
                res = tokenize(row['turn3'], self.version)[0]
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
        """
        Assigns probabilities based on counts
        """
        
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

        old_priors = self.priors

        for row in testing_data:
            emotion = row['label']
            self.true.append(self.emotion2encoding[emotion])

            if self.bag_of_words == 1:
                parsed_message = tokenize(row['turn1'], self.version)[0] + tokenize(row['turn2'], self.version)[0] + tokenize(row['turn3'], self.version)[0]
            else:
                parsed_message = tokenize(row['turn3'], self.version)[0]
            
            if self.weighting_method == 1:
                parsed_message = tokenize(row['turn1'], self.version)[0]
                classification = self.normalize(np.asarray(self.classify(self.ngrams, parsed_message, self.priors, True)))
                for i, emotion in enumerate(self.emotions):
                    self.priors[emotion] = classification[i]
                
                parsed_message = tokenize(row['turn2'], self.version)[0]
                classification = self.normalize(np.asarray(self.classify(self.ngrams, parsed_message, self.priors, True)))
                for i, emotion in enumerate(self.emotions):
                    self.priors[emotion] = classification[i]
        
            classification = self.classify(self.ngrams, parsed_message, self.priors)
            self.pred.append(classification)

            self.priors = old_priors # reset the old priors

        getMetrics(np.array(self.pred), np.array(self.true))

    def normalize(self, arr):
        """
        Normalizes between 0.1 and 1.0

        """
        a = 0.9 * (arr - np.min(arr))/np.ptp(arr) + 0.1
        return a/a.sum(0)

    def classify(self, training_dict, content, priors, return_raw=False):
        """
        Classifies each message according to the trained model

        Parameters:
        training_dict (Object): trained model
        content (String): message to be tested
        priors (Object): priors

        Returns:
        String: classification according to the trained model
        """

        happy = [math.log(priors['happy']), [0,1,0,0]]
        sad = [math.log(priors['sad']), [0,0,1,0]]
        angry = [math.log(priors['angry']), [0,0,0,1]]
        others = [math.log(priors['others']), [1,0,0,0]]
            
        for word in content:
            if word in training_dict:
                happy[0] += float(math.log(training_dict[word]['happy']))
                sad[0] += float(math.log(training_dict[word]['sad'])) 
                angry[0] += float(math.log(training_dict[word]['angry']))
                others[0] += float(math.log(training_dict[word]['others']))
                
        if return_raw: return [happy[0], sad[0], angry[0], others[0]]

        return max([happy, sad, angry, others],key=lambda item:item[0])[1]

    def confusion_matrix(self, normalize=False):
        """
        Computes the confusion matrices for each of the variables
        """

        cn_matrix = confusion_matrix(self.true, self.pred)
        plot_confusion_matrix(cn_matrix, self.emotions, 'Emotions', normalize)
