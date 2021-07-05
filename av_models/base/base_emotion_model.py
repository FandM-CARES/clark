from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class BaseEmotionModel(ClassifierMixin, ABC):
    def __init__(self) -> None:
        self.variables_encoder: LabelEncoder = LabelEncoder().fit(
            [
                "pleasantness",
                "attention",
                "control",
                "certainty",
                "anticipated_effort",
                "responsibility",
            ]
        )
        self.emotions_encoder: LabelEncoder = LabelEncoder().fit(
            [
                "sadness",
                "joy",
                "fear",
                "anger",
                "challenge",
                "boredom",
                "frustration",
            ]
        )

    @classmethod
    def initialize(cls) -> Any:  # Should replace with a Generic soon
        return cls()

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEmotionModel":
        pass

    @abstractmethod
    def predict(self, y: np.ndarray) -> np.ndarray:
        pass
