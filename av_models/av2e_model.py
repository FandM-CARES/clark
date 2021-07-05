import numpy as np
from sklearn.ensemble import RandomForestClassifier

from av_models.base.base_emotion_model import BaseEmotionModel


class AV2EModel(BaseEmotionModel):
    """Appraisal variable to emotion model."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = RandomForestClassifier()

    def fit(self, appraisals: np.ndarray, emotions: np.ndarray) -> np.ndarray:
        """Fit AV2E model."""
        appraisals = np.vstack(appraisals)
        self.classifier.fit(appraisals, emotions.ravel())
        return self

    def predict(self, y: np.ndarray) -> np.ndarray:
        """Predict over sample."""
        return self.classifier.predict(y)
