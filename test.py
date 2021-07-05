from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List, Mapping

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from process_av_data import AVDataset
from utils import MODEL_MAPPING


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--model", "-m", choices=["av2e", "emotion", "av", "clark"], type=str)
    args = parser.parse_args()

    av_dataset = AVDataset.from_json(args.input)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    if args.model in ["av2e", "emotion", "clark"]:
        target = "emotions"
        all_runs: List[float] = []
    elif args.model == "av":
        target = "appraisal_variables"
        all_runs_vars: Mapping[str, List[float]] = defaultdict(list)

    for train_i, test_i in kf.split(av_dataset.text_data):
        text_train, text_test = av_dataset.text_data[train_i], av_dataset.text_data[test_i]
        var_train, var_test = (
            av_dataset.appraisal_variable_data[train_i],
            av_dataset.appraisal_variable_data[test_i],
        )
        em_train, em_test = av_dataset.emotion_data[train_i], av_dataset.emotion_data[test_i]

        model = MODEL_MAPPING[args.model].initialize()
        model.fit(text_train, var_train, em_train)

        source_test = var_test if args.model == "av2e" else text_test

        if target == "appraisal_variables":
            pred = model.predict(source_test)
            for i, var in enumerate(model.variables_encoder.classes_):
                report = classification_report(
                    var_test[:, 2, i],
                    pred[:, i],
                    digits=3,
                    output_dict=True,
                    zero_division=1.0,
                )
                all_runs_vars[var].append(report["accuracy"])

        elif target == "emotions":
            pred = model.predict(source_test)
            report = classification_report(
                em_test[:, 2],
                pred,
                digits=3,
                output_dict=True,
                zero_division=1.0,
            )
            all_runs.append(report["accuracy"])
    if target == "appraisal_variables":
        for var in model.variables_encoder.classes_:
            print(f"{var}: {np.mean(all_runs_vars[var])}")

    elif target == "emotions":
        print(np.mean(all_runs))


if __name__ == "__main__":
    main()
