import numpy as np

from av_models.av2e_model import AV2EModel
from av_models.av_model import AVModel
from av_models.base.base_emotion_model import BaseEmotionModel

EMOTIONS_TO_SEMEVAL_EMOTIONS = {
    "joy": "happy",
    "sadness": "sad",
    "fear": "others",
    "anger": "angry",
    "challenge": "others",
    "boredom": "others",
    "frustration": "others",
}


class CLARKModel(BaseEmotionModel):
    """CLARK Model."""

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.av_model = AVModel()
        self.av2e_model = AV2EModel()

    def _map_to_semeval_emotions(self, emotion: str) -> str:
        return EMOTIONS_TO_SEMEVAL_EMOTIONS[emotion]

    def fit(
        self,
        text: np.ndarray,
        appraisals: np.ndarray,
        emotions: np.ndarray,
        convert_to_semeval: bool = False,
    ) -> "CLARKModel":
        """Fit CLARK model."""
        self.av_model.fit(text, appraisals)

        if convert_to_semeval:
            emotions = np.array(list(map(self._map_to_semeval_emotions, emotions.ravel())))

        self.av2e_model.fit(appraisals, emotions)
        return self

    def predict(self, y: np.ndarray) -> np.ndarray:
        """Predict over samples."""
        av_predictions = self.av_model.predict(y)
        return self.av2e_model.predict(av_predictions)
