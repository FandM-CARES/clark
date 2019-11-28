import math
import statistics

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from models.av_model import AVModel
from nlp_helpers import (determine_pronoun, determine_tense, flatten,
                         ngrams_and_remove_stop_words, normalize,
                         parts_of_speech, tokenize)


class ClarkModel(AVModel):
    """
    TODO: add something here
    """

    def __init__(self, av2e_classifier, ngram_choice):
        super().__init__(ngram_choice)
        self.vectorizer = None
        self.classifiers = list()
        self.av2e_classifier = av2e_classifier

    def fit(self, text, appraisals, emotions):
        """
        """
    
        num_of_appraisals = len(appraisals[0])
        assert num_of_appraisals == 6
        appraisals = np.asarray(appraisals)

        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
        idf_matrix = self.vectorizer.fit_transform(text)

        try:
            for i in range(num_of_appraisals):
                classifier = MultinomialNB()
                classifier.fit(idf_matrix, appraisals[:,i])
                self.classifiers.append(classifier)
        except Exception as e:
            print(f"Could not fit text to appraisals due to the following: {e}")

        try:
            self.av2e_classifier = self.av2e_classifier.fit(appraisals, emotions)
        except Exception as e:
            print(f"Could not fit text to appraisals due to the following: {e}")
        
        return
    
    def predict(self, y):
        """
        """
        
        preds = list()
        for sample in y:
            idf_matrix = self.vectorizer.transform(sample)

            conv = self._predict_appraisal_proba(idf_matrix)
            turn1 = self._predict_appraisal_proba(idf_matrix[0], conv)
            turn3 = self._predict_appraisal_proba(idf_matrix[2], turn1)
            
            pred = [np.argmax(x) for x in turn3]

            preds.append(pred)

        return self.av2e_classifier.predict(preds)

    def _predict_appraisal_proba(self, X, priors=[]) -> list:
        """
        """

        appraisal_pred = list()
        for i in range(6): #dependent on hardcoded number of appraisals (bad)
            res = self.classifiers[i].predict_log_proba(X)
            appraisal_pred.append(res[0])
        
        if len(priors) > 0:
            return np.add(appraisal_pred, priors)
        return appraisal_pred