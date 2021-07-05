from argparse import ArgumentParser
from pathlib import Path

from process_av_data import AVDataset
from utils import MODEL_MAPPING, save_model


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--format", choices=["csv", "json"], type=str, default="json")
    parser.add_argument("--model", "-m", choices=["av2e", "av", "clark"], type=str)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-on-3", action="store_true", default=False)
    args = parser.parse_args()

    av_dataset = AVDataset.from_json(args.input)
    model = MODEL_MAPPING[args.model].initialize()
    if args.train_on_3:
        text, appraisals, emotions = (
            av_dataset.text_data[:, 2, :],
            av_dataset.appraisal_variable_data[:, 2, :],
            av_dataset.emotion_data[:, 2],
        )
    model.fit(text, appraisals, emotions, convert_to_semeval=True)

    save_model(model, args.output)
    print("Model done training.")


if __name__ == "__main__":
    main()
