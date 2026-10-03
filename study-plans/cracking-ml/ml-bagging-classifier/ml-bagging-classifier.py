from fractions import Fraction
import numpy as np

def bagging_classify(X_train: list, y_train: list, X_test: list, n_estimators: int = 10, max_depth: int = 5, seed: int = 42) -> list:
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)
    rng = np.random.RandomState(seed)

    def gini(values):
        _, counts = np.unique(values, return_counts=True)
        return Fraction(1) - sum((Fraction(int(count), len(values)) ** 2 for count in counts), Fraction(0))

    def leaf(values):
        classes, counts = np.unique(values, return_counts=True)
        return {"leaf": True, "value": classes[np.argmax(counts)]}

    def build(X, values, depth):
        if depth >= max_depth or len(values) < 2 or len(np.unique(values)) == 1:
            return leaf(values)
        feature_count = X.shape[1]
        features = np.arange(X.shape[1])
        parent = gini(values)
        best_gain, best_feature, best_threshold = Fraction(0), None, None
        for feature in features:
            for threshold in np.unique(X[:, feature]):
                left = X[:, feature] <= threshold
                if not np.any(left) or np.all(left):
                    continue
                gain = parent - Fraction(int(np.sum(left)), len(values)) * gini(values[left]) - Fraction(int(np.sum(~left)), len(values)) * gini(values[~left])
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, int(feature), threshold
        if best_feature is None:
            return leaf(values)
        left = X[:, best_feature] <= best_threshold
        return {"leaf": False, "feature": best_feature, "threshold": best_threshold,
                "left": build(X[left], values[left], depth + 1),
                "right": build(X[~left], values[~left], depth + 1)}

    def predict_one(node, row):
        while not node["leaf"]:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]

    trees = []
    for _ in range(n_estimators):
        indices = rng.randint(0, len(X_train), size=len(X_train))
        trees.append(build(X_train[indices], y_train[indices], 0))
    predictions = []
    for row in X_test:
        classes, counts = np.unique([predict_one(tree, row) for tree in trees], return_counts=True)
        predictions.append(int(classes[np.argmax(counts)]))
    return predictions
