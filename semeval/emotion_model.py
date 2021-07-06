from nltk.tokenize import casual_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

SEMEVAL_EMOTION_LABELS = ["angry", "happy", "sad", "others"]


class EmotionModel:
    """CLARK emotion model from SemEval-2019 paper."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", tokenizer=casual_tokenize, max_df=0.8
        )
        self.predictor = MultinomialNB()
        self.label_encoder = LabelEncoder()

    def fit(self, texts: np.ndarray, emotions: np.ndarray) -> "EmotionModel":
        """Fit emotion model."""
        self.label_encoder.fit(SEMEVAL_EMOTION_LABELS)
        idf_matrix = self.vectorizer.fit_transform([conv.ravel()[2] for conv in texts])
        self.predictor = self.predictor.fit(idf_matrix, emotions.ravel())
        return self

    def predict(self, texts: np.ndarray) -> np.ndarray:
        """Predict over sample."""
        preds = []
        for sample in tqdm(texts):
            idf_matrix = self.vectorizer.transform(sample)
            conv = np.mean(self.predictor.predict_proba(idf_matrix), axis=0)
            turn3 = self.predictor.predict_proba(idf_matrix[2]) + conv
            res = self.label_encoder.inverse_transform([np.argmax(turn3)])
            preds.append(res)
        return np.array(preds)
