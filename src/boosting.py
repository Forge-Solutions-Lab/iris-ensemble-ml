import numpy as np
from sklearn.utils.validation import has_fit_parameter

# Keeps alpha finite when a model makes no weighted error on the training set.
MIN_ERROR = 1e-10
# alpha = 0.5 * ln((1 - e) / e) is zero or negative from this error upward.
MAX_ERROR = 0.5


def _fit_weighted(model, X_train, y_train, weights, rng):
    if has_fit_parameter(model, "sample_weight"):
        return model.fit(X_train, y_train, sample_weight=weights)
    # k-NN and MLP cannot take sample weights, so draw a training set in proportion to them.
    resampled = rng.choice(len(X_train), size=len(X_train), p=weights)
    return model.fit(X_train[resampled], y_train[resampled])


def run_boosting(models, X_train, y_train, X_test, seed):
    rng = np.random.default_rng(seed)
    weights = np.full(len(X_train), 1 / len(X_train))
    class_scores = np.zeros((len(np.unique(y_train)), len(X_test)))
    rounds = []
    for name, model in models.items():
        _fit_weighted(model, X_train, y_train, weights, rng)
        misclassified = model.predict(X_train) != y_train
        error = max(weights[misclassified].sum(), MIN_ERROR)
        if error >= MAX_ERROR:
            rounds.append({"model": name, "error": float(error), "alpha": 0.0, "skipped": True})
            continue
        alpha = 0.5 * np.log((1 - error) / error)
        weights = weights * np.exp(np.where(misclassified, alpha, -alpha))
        weights /= weights.sum()
        class_scores[model.predict(X_test), np.arange(len(X_test))] += alpha
        rounds.append({"model": name, "error": float(error), "alpha": float(alpha), "skipped": False})
    return class_scores.argmax(axis=0), rounds
