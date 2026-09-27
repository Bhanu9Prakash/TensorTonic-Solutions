import numpy as np

def permutation_importance(X: list, y: list, predict_fn, n_repeats: int = 5, seed: int = 42) -> list:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    baseline = np.mean(predict_fn(X) == y)
    importances = []
    for feature in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            permuted = X.copy()
            permuted[:, feature] = rng.permutation(permuted[:, feature])
            drops.append(baseline - np.mean(predict_fn(permuted) == y))
        importances.append(round(float(np.mean(drops)), 4))
    return importances
