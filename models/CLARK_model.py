import math

import numpy as np

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
        self.true = []
        self.pred = []
        self.micro_fscores = 0.0
        self.macro_fscores = 0.0
        self.av2e_classifier = av2e_classifier

    def train(self, training_data):
        self.__build_av2e_classifier(training_data)

        for var in self.variables:
            self._train_by_variable(training_data, var)

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

            tokenized_turn1 = tokenize(row["turn1"]["text"])
            tokenized_turn2 = tokenize(row["turn2"]["text"])
            tokenized_turn3 = tokenize(row["turn3"]["text"])

            conv = tokenized_turn1 + tokenized_turn2 + tokenized_turn3

            parsed_message = flatten([ngrams_and_remove_stop_words(x, self.ngram_choice) for x in [
                                     tokenized_turn1, tokenized_turn2, tokenized_turn3]])
            for var in self.variables:
                classification = normalize(self._classify(
                    self.ngrams[var], parsed_message, conv, u_priors[var], var))
                for i, e in enumerate(self.variable_dimensions):
                    u_priors[var][e] = classification[i]

            parsed_message = ngrams_and_remove_stop_words(
                tokenized_turn1, self.ngram_choice)
            for var in self.variables:
                classification = normalize(self._classify(
                    self.ngrams[var], parsed_message, tokenized_turn1, u_priors[var], var))
                for i, e in enumerate(self.variable_dimensions):
                    u_priors[var][e] = classification[i]

            parsed_message = ngrams_and_remove_stop_words(
                tokenized_turn3, self.ngram_choice)
            var_classification = {dim: "" for dim in self.variables}
            for var in self.variables:
                var_classification[var] = self._classify(
                    self.ngrams[var], parsed_message, tokenized_turn3, u_priors[var], var, False)

            self.true.append(row["turn3"]["emotion"])
            emo_class = self.__map_to_emotion(var_classification)
            self.pred.append(emo_class[0])

        super(AVModel, self).calculate_scores()

    def __build_av2e_classifier(self, data):
        X = [list(d["turn3"]["appraisals"].values()) for d in data]
        y = [em["turn3"]["emotion"] for em in data]

        self.av2e_classifier = self.av2e_classifier.fit(X, y)

    def __map_to_emotion(self, variables):
        v = [self.variable_dimensions.index(x)
             for x in list(variables.values())]
        return self.av2e_classifier.predict(np.asarray(v).reshape(1, -1))
