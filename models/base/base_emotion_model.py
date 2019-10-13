from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import confusion_matrix

from graphing_helpers import plot_confusion_matrix

np.seterr(all="raise")


class BaseEmotionModel(ABC):
    def __init__(self):
        self.variables = ["pleasantness", "attention", "control",
                          "certainty", "anticipated_effort", "responsibility"]
        self.emotions = ["sadness", "joy", "fear",
                         "anger", "challenge", "boredom", "frustration"]
        self.true = list()
        self.pred = list()
        self.micro_scores = dict()
        self.macro_scores = dict()

    @abstractmethod
    def train(self, training_data: list) -> None:
        """
        Builds a trained model

        Parameters:
        training_data (array): training data used to train the model

        Returns:
        None
        """
        pass

    @abstractmethod
    def test(self, testing_data: list) -> None:
        pass

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
                                np.logical_or(np.logical_and(self.pred == "sadness", self.true == "sadness"), np.logical_and(
                                    self.pred == "joy", self.true == "joy")),
                                np.logical_and(self.pred == "fear", self.true == "fear")),
                            np.logical_and(self.pred == "anger", self.true == "anger")),
                        np.logical_and(self.pred == "challenge", self.true == "challenge")),
                    np.logical_and(self.pred == "boredom", self.true == "boredom")),
                np.logical_and(self.pred == "frustration", self.true == "frustration")))
        tp_fp = len(self.pred)
        tp_fn = len(self.true)

        pi = tp / tp_fp
        ro = tp / tp_fn
        try:
            self.micro_fscores = 2 * pi * ro / (pi + ro)
        except ZeroDivisionError:
            self.micro_fscores = 0.0

        temp_macro = 0
        for e in self.emotions:
            tp_e = np.sum(np.logical_and(self.pred == e, self.true == e))
            tp_fp_e = len([x for x in self.pred if x != e])
            tp_fn_e = len([x for x in self.true if x == e])

            try:
                pi_e = tp_e / tp_fp_e
            except (ZeroDivisionError, FloatingPointError):
                pi_e = 0.0

            try:
                ro_e = tp_e / tp_fn_e
            except (ZeroDivisionError, FloatingPointError):
                ro_e = 0.0

            try:
                temp_macro += 2 * pi_e * ro_e / (pi_e + ro_e)
            except (ZeroDivisionError, FloatingPointError):
                temp_macro += 0.0

        self.macro_fscores = temp_macro / 7

    def confusion_matrix(self, title, normalized=False):
        """
        Computes the confusion matrices for each of the variables
        """

        cn_matrix = confusion_matrix(self.true, self.pred)
        plot_confusion_matrix(cn_matrix, self.emotions,
                              title, normalized)
