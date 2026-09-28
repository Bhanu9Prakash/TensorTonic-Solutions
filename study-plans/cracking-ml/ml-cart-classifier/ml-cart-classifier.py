from fractions import Fraction
import numpy as np

def cart_classify(X_train: list, y_train: list, X_test: list, max_depth: int = 5, min_samples: int = 2) -> list:
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)

    def impurity(values):
        _, counts = np.unique(values, return_counts=True)
        return 1 - sum(Fraction(int(count) ** 2, len(values) ** 2) for count in counts)

    def leaf(values):
        classes, counts = np.unique(values, return_counts=True)
        return {"leaf": True, "value": classes[np.argmax(counts)]}

    def build(X, values, depth):
        if depth >= max_depth or len(values) < min_samples or len(np.unique(values)) == 1:
            return leaf(values)
        parent = impurity(values)
        best_gain, best_feature, best_threshold = 0.0, None, None
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left = X[:, feature] <= threshold
                if not np.any(left) or np.all(left):
                    continue
                left_weight = Fraction(int(np.sum(left)), len(values))
                gain = parent - left_weight * impurity(values[left]) - (1 - left_weight) * impurity(values[~left])
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
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

    tree = build(X_train, y_train, 0)
    return [int(predict_one(tree, row)) for row in X_test]
