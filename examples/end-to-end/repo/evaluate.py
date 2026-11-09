# evaluate.py
#
# Evaluation script for ML model development.
# Trains the model and evaluates accuracy on a validation split.

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class IncorrectSubmissionError(Exception):
    """Raised when submission format is incorrect."""
    pass


def evaluate_for_accuracy(
    submission_df: pd.DataFrame, 
    answers_df: pd.DataFrame, 
    target_column: str = "Transported", 
    id_column: str = "PassengerId"
) -> float:
    """
    Evaluate submission accuracy against ground truth.
    
    Args:
        submission_df: DataFrame with predictions
        answers_df: DataFrame with ground truth
        target_column: Name of the target column
        id_column: Name of the ID column
    
    Returns:
        Accuracy score between 0.0 and 1.0
    """
    # Answers checks
    assert target_column in answers_df.columns, f"Answers must have a `{target_column}` column"
    assert id_column in answers_df.columns, f"Answers must have a `{id_column}` column"

    # Submission checks
    if len(submission_df) != len(answers_df):
        raise IncorrectSubmissionError("Submission must have the same length as the answers.")
    if target_column not in submission_df.columns:
        raise IncorrectSubmissionError(f"Submission must have a `{target_column}` column")
    if id_column not in submission_df.columns:
        raise IncorrectSubmissionError(f"Submission must have a `{id_column}` column")

    # Sort on id to ensure correct ordering
    submission_df = submission_df.sort_values(by=id_column)
    answers_df = answers_df.sort_values(by=id_column)

    if (submission_df[id_column].values != answers_df[id_column].values).any():
        raise IncorrectSubmissionError(f"Submission and Answers `{id_column}`'s do not match")

    return accuracy_score(submission_df[target_column], answers_df[target_column])


def read_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read and split the data.
    
    Args:
        data_dir: Path to directory containing train.csv and test.csv
    
    Returns:
        Tuple of (train_df, validation_df, test_df)
    """
    train_df = pd.read_csv(data_dir / "train.csv")
    # Split training data into train and validation sets
    train_df, validation_df = train_test_split(train_df, test_size=0.1, random_state=0)
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, validation_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ML model")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Load data
    train_df, validation_df, test_df = read_data(args.data_dir)

    # Import training and prediction functions
    from train import train_model, predict_with_model

    # Validate that required functions exist and are callable
    assert callable(train_model), "train_model function must exist and be callable"
    assert callable(predict_with_model), "predict_with_model function must exist and be callable"

    # Step 1: Train the model
    print("Training model...")
    model = train_model(train_df, args.seed)

    # Step 2: Generate predictions on validation set
    print("Generating validation predictions...")
    validation_submission_df = predict_with_model(model, validation_df)

    # Step 3: Evaluate accuracy on validation set
    acc = evaluate_for_accuracy(validation_submission_df, validation_df)
    print(f"accuracy: {acc:.6f}")

    # Step 4: Generate predictions on test set
    print("Generating test predictions...")
    test_submission_df = predict_with_model(model, test_df)
    test_submission_df.to_csv("submission.csv", index=False)
    print("Test predictions saved to submission.csv")
    
    # Output score for Kapso
    print(f"\n__SCORE__: {acc}")
