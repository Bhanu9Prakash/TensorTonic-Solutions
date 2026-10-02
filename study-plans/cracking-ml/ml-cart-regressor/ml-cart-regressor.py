from fractions import Fraction
import numpy as np

def cart_regress(
    X_train: list,
    y_train: list,
    X_test: list,
    max_depth: int = 5,
    min_samples: int = 2
) -> list:

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    def calculate_mse(values):
        # Use exact arithmetic to avoid floating-point tie errors
        values = [Fraction(float(value)) for value in values]

        mean = sum(values) / len(values)

        return sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

    def best_split(X, y):

        features = X.shape[1]

        best_score = None
        best_feature = None
        best_threshold = None

        for feature in range(features):

            # Use actual feature values as thresholds
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:

                left = X[:, feature] <= threshold
                right = ~left

                # Skip invalid splits
                if not np.any(left) or not np.any(right):
                    continue

                mse_left = calculate_mse(y[left])
                mse_right = calculate_mse(y[right])

                left_weight = Fraction(
                    int(np.sum(left)),
                    len(y)
                )

                right_weight = 1 - left_weight

                score = (
                    left_weight * mse_left
                    + right_weight * mse_right
                )

                # < keeps the first split when there is a tie
                if best_score is None or score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(X, y, depth=0):

        prediction = np.mean(y)

        if (
            depth >= max_depth
            or len(y) < min_samples
            or np.all(y == y[0])
        ):
            return {
                "prediction": prediction
            }

        feature, threshold = best_split(X, y)

        # No valid split
        if feature is None:
            return {
                "prediction": prediction
            }

        left = X[:, feature] <= threshold
        right = ~left

        left_tree = build_tree(
            X[left],
            y[left],
            depth + 1
        )

        right_tree = build_tree(
            X[right],
            y[right],
            depth + 1
        )

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree
        }

    def predict_one(X, tree):

        if "prediction" in tree:
            return tree["prediction"]

        feature = tree["feature"]
        threshold = tree["threshold"]

        if X[feature] <= threshold:
            return predict_one(
                X,
                tree["left"]
            )

        return predict_one(
            X,
            tree["right"]
        )

    tree = build_tree(
        X_train,
        y_train
    )

    predictions = []

    for row in X_test:
        predictions.append(
            round(
                float(predict_one(row, tree)),
                4
            )
        )

    return predictions