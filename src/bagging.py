from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np


def majority_vote(prediction_matrix, class_count):
    vote_counts = np.apply_along_axis(np.bincount, 0, prediction_matrix, minlength=class_count)
    # argmax returns the first maximum, so ties resolve to the lowest class index.
    return vote_counts.argmax(axis=0)


def _fit_predict(model, sample_indices, X_train, y_train, X_test):
    model.fit(X_train[sample_indices], y_train[sample_indices])
    return model.predict(X_test)


def run_bagging(models, X_train, y_train, X_test, seed, parallel=True):
    rng = np.random.default_rng(seed)
    sample_count = len(X_train)
    bootstrap_indices = [rng.integers(0, sample_count, sample_count) for _ in models]
    fit_predict = partial(_fit_predict, X_train=X_train, y_train=y_train, X_test=X_test)
    if parallel:
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            predictions = list(executor.map(fit_predict, models.values(), bootstrap_indices))
    else:
        predictions = list(map(fit_predict, models.values(), bootstrap_indices))
    prediction_matrix = np.array(predictions)
    return majority_vote(prediction_matrix, len(np.unique(y_train))), prediction_matrix
