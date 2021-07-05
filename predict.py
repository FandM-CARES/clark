from argparse import ArgumentParser
from pathlib import Path

from sklearn.metrics import classification_report

from process_av_data import AVDataset
from utils import load_model


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--format", choices=["csv", "json"], type=str)
    parser.add_argument("--saved-model", "-sm", type=Path)
    args = parser.parse_args()

    av_dataset = AVDataset.from_json(args.input)
    print("Finished processing data.")

    model = load_model(args.saved_model)
    print("Loaded model.")

    pred_emotions = model.predict(av_dataset.text_data)

    report = classification_report(av_dataset.emotion_data, pred_emotions)
    print(report)


if __name__ == "__main__":
    main()
