# train.py
#
# ML model training script for Spaceship Titanic competition.
# This file will be optimized by Kapso.

# ============================================================================
# CRITICAL: DO NOT CHANGE FUNCTION NAMES OR SIGNATURES
# ============================================================================
# The following functions are part of the stable API contract:
# - train_model(train_df: pd.DataFrame, random_state: int = 0)
# - predict_with_model(model, test_df: pd.DataFrame) -> pd.DataFrame
#
# These function names and signatures MUST remain unchanged as they are
# imported and called by evaluate.py. Only modify the internal implementation.
# ============================================================================

import pandas as pd
from sklearn.dummy import DummyClassifier


def train_model(train_df: pd.DataFrame, random_state: int = 0):
    """
    Train a model on the training data and return it.
    This function will be optimized by Kapso.

    IMPORTANT: This function name and signature must NOT be changed.
    Only the internal implementation should be modified.

    Args:
        train_df: Training DataFrame with features and target
        random_state: Random seed for reproducibility

    Returns:
        Trained model object
    """
    # Make local copy to prevent modifying DataFrame in the caller's scope
    train_df_local = train_df.copy()

    # --- Stage 1: Prepare training data ---
    # Separate target variable from training data
    y_train_series = train_df_local["Transported"]
    # Features for training: drop target and PassengerId
    X_train_df = train_df_local.drop(columns=["Transported", "PassengerId"])

    # --- Stage 2: Preprocessing and Model Training ---
    # This block will be optimized by Kapso for:
    # - Imputation (e.g., SimpleImputer)
    # - Scaling (e.g., StandardScaler)
    # - Encoding categorical features (e.g., OneHotEncoder, LabelEncoder)
    # - Feature Engineering (creating new features)
    # - Model Selection (e.g., RandomForestClassifier, XGBoost, LightGBM)
    # - Hyperparameter Tuning (e.g., GridSearchCV, RandomizedSearchCV)
    # - Using sklearn.pipeline.Pipeline for robust preprocessing

    # --- Baseline: Simple DummyClassifier (Kapso will replace this) ---
    model = DummyClassifier(strategy="most_frequent", random_state=random_state)
    model.fit(X_train_df, y_train_series)
    # --- End of Optimizable Block ---

    return model


def predict_with_model(model, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions on test data using a trained model.
    This function should remain relatively stable.

    IMPORTANT: This function name and signature must NOT be changed.
    Only the internal implementation should be modified.

    Args:
        model: Trained model object
        test_df: Test DataFrame with features (and possibly target for validation)

    Returns:
        DataFrame with PassengerId and predictions
    """
    # Make local copy to prevent modifying DataFrame in the caller's scope
    test_df_local = test_df.copy()

    # Preserve PassengerId for the submission file
    passenger_ids = test_df_local["PassengerId"].copy()

    # Features for prediction: drop target (if present) and PassengerId
    X_test_df = test_df_local.drop(columns=["Transported", "PassengerId"], errors="ignore")

    # Make predictions
    predictions = model.predict(X_test_df)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        "PassengerId": passenger_ids, 
        "Transported": predictions.astype(bool)
    })

    return submission_df
