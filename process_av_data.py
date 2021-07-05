import json
from pathlib import Path
from typing import Any, List, Mapping

import numpy as np


class AVDataset:
    def __init__(
        self, text_data: np.ndarray, appraisal_variable_data: np.ndarray, emotion_data: np.ndarray
    ) -> None:
        self.text_data = text_data
        self.appraisal_variable_data = appraisal_variable_data
        self.emotion_data = emotion_data

    @staticmethod
    def from_json(json_file: Path, convos_file: Path = Path("data/all_convos.json")) -> "AVDataset":
        """Parse JSON file."""

        def _combine_iterances(data: List[Mapping[str, Any]]) -> "AVDataset":
            """Combine iterances from JSON files."""
            text_samples, appraisal_samples, emotion_samples = [], [], []
            for sample in data:
                curr_text_samples, curr_appraisal_samples, curr_emotion_samples = [], [], []
                for turn in ["turn1", "turn2", "turn3"]:
                    curr_text_samples.append([sample[turn]["text"]])
                    curr_appraisal_samples.append(
                        [int(val) for val in sample[turn]["appraisals"].values()]
                    )
                    curr_emotion_samples.append(sample[turn]["emotion"])
                text_samples.append(curr_text_samples)
                appraisal_samples.append(curr_appraisal_samples)
                emotion_samples.append(curr_emotion_samples)

            return AVDataset(
                np.array(text_samples), np.array(appraisal_samples), np.array(emotion_samples)
            )

        with open(convos_file, "r") as handle:
            convos = json.load(handle)

        parsed = []
        with open(json_file, encoding="utf-8") as raw_data:
            data = json.load(raw_data)
            for val in data:
                for conv in convos:
                    if conv["id"] == val["id"]:
                        val["turn1"]["text"] = conv["turn1"]
                        val["turn2"]["text"] = conv["turn2"]
                        val["turn3"]["text"] = conv["turn3"]
                        parsed.append(val)
                        continue
        return _combine_iterances(parsed)
