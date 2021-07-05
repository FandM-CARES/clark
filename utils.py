from pathlib import Path
import pickle
from typing import Mapping

from av_models.av2e_model import AV2EModel
from av_models.av_model import AVModel
from av_models.base.base_emotion_model import BaseEmotionModel
from av_models.clark_model import CLARKModel

MODEL_MAPPING: Mapping[str, BaseEmotionModel] = {
    "av": AVModel,
    "av2e": AV2EModel,
    "clark": CLARKModel,
}


def load_model(saved_model: Path) -> BaseEmotionModel:
    with open(saved_model, "rb") as handle:
        return pickle.load(handle)


def save_model(model: BaseEmotionModel, output_path: Path) -> None:
    with open(output_path, "wb+") as handle:
        pickle.dump(model, handle)
