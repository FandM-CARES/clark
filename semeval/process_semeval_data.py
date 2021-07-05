import csv
from pathlib import Path

import numpy as np


class SemEvalDataset:
    def __init__(self, text_data: np.ndarray, emotion_data: np.ndarray) -> None:
        self.text_data = text_data
        self.emotion_data = emotion_data

    @staticmethod
    def from_csv(input_file: Path) -> "SemEvalDataset":
        """Parse CSV file into SemEvalDataset."""
        text_samples, emotion_samples = [], []

        with open(input_file, encoding="utf-8") as raw_data:
            csv_reader = csv.reader(
                raw_data,
                delimiter=",",
            )
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue
                curr_text_samples = [row[1], row[2], row[3]]
                text_samples.append(curr_text_samples)
                emotion_samples.append(row[4])

        return SemEvalDataset(np.array(text_samples), np.array(emotion_samples))
