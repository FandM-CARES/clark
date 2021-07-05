from typing import List

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import tqdm

from av_models.base.base_emotion_model import BaseEmotionModel


class AVModel(BaseEmotionModel):
    """Appraisal Variables Model."""

    def __init__(self) -> None:
        super().__init__()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_df=0.8)
        self.classifiers: List[ClassifierMixin] = []

    def fit(self, text: np.ndarray, appraisals: np.ndarray) -> np.ndarray:
        """Fit AV model."""
        idf_matrix = self.vectorizer.fit_transform(text.ravel())

        appraisals = np.vstack(appraisals)
        for i in range(len(self.variables_encoder.classes_)):
            classifier = MultinomialNB()
            classifier.fit(idf_matrix, appraisals[:, i])
            self.classifiers.append(classifier)

        return self

    def predict(self, y: np.ndarray) -> np.ndarray:
        """Predict over sample."""
        preds = []
        for sample in tqdm.tqdm(y):
            idf_matrix = self.vectorizer.transform(sample.ravel())

            turn1 = self._predict_appraisal_proba(idf_matrix[0])
            turn3 = self._predict_appraisal_proba(idf_matrix[2], turn1)

            preds.append([np.argmax(x) for x in turn3])

        return np.array(preds)

    def _predict_appraisal_proba(
        self, X: np.ndarray, priors: np.ndarray = np.array([])
    ) -> np.ndarray:
        """Predict appraisal probabilities."""
        appraisal_pred = []
        for i in range(len(self.variables_encoder.classes_)):
            res = self.classifiers[i].predict_proba(X)
            appraisal_pred.append(res)

        if len(priors) > 0:
            return np.array(appraisal_pred) + priors
        return np.array(appraisal_pred)
