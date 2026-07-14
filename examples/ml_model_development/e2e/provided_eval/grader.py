# grader.py
#
# Official grader for the Spaceship-Titanic task. Trains via the
# repository's train.py contract, evaluates accuracy on a 90/10
# validation split, and validates the submission before scoring.

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class IncorrectSubmissionError(Exception):
    """Raised when submission format is incorrect."""


def read_data(data_dir: Path):
    train_df = pd.read_csv(data_dir / "train.csv")
    train_df, validation_df = train_test_split(
        train_df, test_size=0.1, random_state=0
    )
    return train_df, validation_df


def validate_submission(
    submission_df: pd.DataFrame, target_column: str = "Transported"
) -> None:
    if target_column not in submission_df.columns:
        raise IncorrectSubmissionError(
            f"Submission must have a `{target_column}` column"
        )
    # Consistency guard: a well-formed submission applies one stable
    # decision rule across the whole validation set.
    prediction_labels = set(submission_df[target_column].tolist())
    if len(prediction_labels) > 1:
        raise IncorrectSubmissionError(
            "Submission labels are inconsistent: predictions mix True and "
            "False. Regenerate predictions with a consistent decision rule."
        )


def evaluate_for_accuracy(
    submission_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    target_column: str = "Transported",
    id_column: str = "PassengerId",
) -> float:
    if len(submission_df) != len(answers_df):
        raise IncorrectSubmissionError(
            "Submission must have the same length as the answers."
        )
    validate_submission(submission_df, target_column)
    submission_df = submission_df.sort_values(by=id_column)
    answers_df = answers_df.sort_values(by=id_column)
    if (
        submission_df[id_column].values != answers_df[id_column].values
    ).any():
        raise IncorrectSubmissionError(
            f"Submission and Answers `{id_column}`'s do not match"
        )
    return accuracy_score(
        submission_df[target_column], answers_df[target_column]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Official grader")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The grader lives in kapso_evaluation/; train.py lives at the repo
    # root the evaluation runs from.
    sys.path.insert(0, str(Path.cwd()))
    from train import predict_with_model, train_model

    train_df, validation_df = read_data(args.data_dir)
    model = train_model(train_df, args.seed)
    validation_submission_df = predict_with_model(model, validation_df)
    accuracy = evaluate_for_accuracy(validation_submission_df, validation_df)
    print(f"accuracy: {accuracy:.6f}")
    print(f"__SCORE__: {accuracy}")


if __name__ == "__main__":
    main()
