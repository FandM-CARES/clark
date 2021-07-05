from argparse import ArgumentParser
from pathlib import Path

from scipy.sparse import data
from semeval.process_semeval_data import SemEvalDataset

from utils import save_model

from semeval.emotion_model import EmotionModel


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = SemEvalDataset.from_csv(args.input)
    model = EmotionModel()
    model.fit(dataset.text_data, dataset.emotion_data)

    save_model(model, args.output)
    print("Model done training.")


if __name__ == "__main__":
    main()
