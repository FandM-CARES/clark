from argparse import ArgumentParser
from pathlib import Path
from semeval.process_semeval_data import SemEvalDataset

from sklearn.metrics import classification_report

from utils import load_model


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--saved-model", "-sm", type=Path)
    args = parser.parse_args()

    dataset = SemEvalDataset(args.input)
    print("Finished processing data.")

    model = load_model(args.saved_model)
    print("Loaded model.")

    print("Predicting...")
    pred_emotions = model.predict(dataset.text_data)

    report = classification_report(dataset.emotion_data, pred_emotions)
    print(report)


if __name__ == "__main__":
    main()
