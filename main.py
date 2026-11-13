import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.bagging import run_bagging
from src.boosting import run_boosting
from src.dataset import load_dataset, split_and_scale
from src.models import build_models

RANDOM_STATE = 42
# Single millisecond timings are noisy, so each method is timed several times and the median is kept.
TIMING_REPEATS = 10
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "iris.csv"
RESULTS_PATH = PROJECT_DIR / "output" / "results.json"


def measure(run):
    durations_ms = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        outcome = run()
        durations_ms.append((time.perf_counter() - start) * 1000)
    return outcome, float(np.median(durations_ms))


def main():
    features, labels, class_names = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = split_and_scale(features, labels, RANDOM_STATE)
    print(f"Dataset: {features.shape[0]} rows, {features.shape[1]} features, classes {class_names}")
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

    results = []
    for name in build_models(RANDOM_STATE):
        predictions, time_ms = measure(
            lambda: build_models(RANDOM_STATE)[name].fit(X_train, y_train).predict(X_test)
        )
        results.append({"method": name, "accuracy": accuracy_score(y_test, predictions), "time_ms": time_ms})

    for method, parallel in [("Bagging (parallel)", True), ("Bagging (sequential)", False)]:
        (predictions, prediction_matrix), time_ms = measure(
            lambda: run_bagging(build_models(RANDOM_STATE), X_train, y_train, X_test, RANDOM_STATE, parallel)
        )
        results.append({"method": method, "accuracy": accuracy_score(y_test, predictions), "time_ms": time_ms})

    (predictions, boosting_rounds), time_ms = measure(
        lambda: run_boosting(build_models(RANDOM_STATE), X_train, y_train, X_test, RANDOM_STATE)
    )
    results.append({"method": "Boosting (sequential)", "accuracy": accuracy_score(y_test, predictions), "time_ms": time_ms})

    results_table = pd.DataFrame(results).set_index("method")
    results_table["accuracy"] *= 100
    print(f"Bagging prediction matrix shape: {prediction_matrix.shape}\n")
    print("Boosting rounds:")
    print(pd.DataFrame(boosting_rounds).to_string(index=False, float_format="%.4f"), "\n")
    print(f"Accuracy (%) and median time (ms) over {TIMING_REPEATS} runs:")
    print(results_table.to_string(float_format="%.2f"))

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "dataset": {"rows": int(features.shape[0]), "train_rows": len(X_train), "test_rows": len(X_test),
                    "classes": class_names},
        "timing_repeats": TIMING_REPEATS,
        "bagging_prediction_matrix_shape": list(prediction_matrix.shape),
        "boosting_rounds": boosting_rounds,
        "results": results,
    }, indent=2))
    print(f"\nSaved {RESULTS_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
