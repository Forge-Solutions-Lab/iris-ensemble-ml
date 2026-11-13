import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["sepal-length", "sepal-width", "petal-length", "petal-width"]
LABEL_COLUMN = "species"
EXPECTED_CLASS_COUNT = 3
TEST_SIZE = 0.2


def load_dataset(path):
    frame = pd.read_csv(path)
    missing_columns = set(FEATURE_COLUMNS + [LABEL_COLUMN]) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    if frame[FEATURE_COLUMNS + [LABEL_COLUMN]].isna().any().any():
        raise ValueError(f"{path} contains missing values")
    class_names, labels = np.unique(frame[LABEL_COLUMN], return_inverse=True)
    if len(class_names) != EXPECTED_CLASS_COUNT:
        raise ValueError(f"Expected {EXPECTED_CLASS_COUNT} classes, found {len(class_names)}")
    return frame[FEATURE_COLUMNS].to_numpy(dtype=float), labels, class_names.tolist()


def split_and_scale(features, labels, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, stratify=labels, random_state=seed
    )
    # Fit on the training split only so no test statistics leak into the models.
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test
