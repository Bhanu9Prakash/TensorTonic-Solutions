import numpy as np

def random_forest_classify(X_train: list, y_train: list, X_test: list, n_estimators: int = 10, max_depth: int = 5, max_features = "sqrt", seed: int = 42) -> list:
    """
    Returns one random-forest prediction per test row.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    rng = np.random.RandomState(seed)
    n_samples, n_features = X_train.shape
    

    if max_features == "sqrt":
        n_split_features = max(1, int(np.sqrt(n_features)))
    elif max_features == "log2":
        n_split_features = max(1, int(np.log2(n_features)))
    elif max_features is None:
        n_split_features = n_features
    else:
        n_split_features = int(max_features)


    n_split_features = min(n_split_features, n_features)

    def gini(y):
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts = True)
        probabilities = counts/len(y)

        return 1 - np.sum(probabilities**2)

    def majority_class(y):
        classes, counts = np.unique(y, return_counts = True)

        return classes[np.argmax(counts)].item()

    def find_best_split(X, y):
        feature_indices = rng.choice(X.shape[1], size = n_split_features, replace = False)
        best_score, best_feature, best_threshold = float("inf"), None, None
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = X[:, feature] <= threshold
                right = ~left

                if not np.any(left) or not np.any(right):
                    continue
                
                left_gini = gini(y[left])
                right_gini = gini(y[right])
                score = ((len(y[left])/len(y))*left_gini) + ((len(y[right])/len(y))*right_gini)

                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold


    def build_tree(X, y, depth = 0):
        prediction = majority_class(y)
        if len(np.unique(y)) == 1 or depth >= max_depth:
            return {"prediction" : prediction}

        feature, threshold = find_best_split(X, y)

        if feature is None:
            return {"prediction": prediction}

        left = X[:, feature] <= threshold
        right = ~left

        
        left_tree = build_tree(X[left], y[left], depth + 1)
        right_tree = build_tree(X[right], y[right], depth + 1)

        return {"feature": feature, "threshold": threshold, "left": left_tree, "right": right_tree}

    def predict_one(X, tree):
        if "prediction" in tree:
            return tree["prediction"]
        feature, threshold = tree["feature"], tree["threshold"]
        if X[feature] <= threshold:
            return predict_one(X, tree["left"])
        return predict_one(X, tree["right"])

    forest = []

    for _ in range(n_estimators):
        bootstrap_indices = rng.choice(n_samples, size = n_samples, replace = True)
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]

        tree = build_tree(X_bootstrap, y_bootstrap)

        forest.append(tree)

    final_predictions = []

    for row in X_test:
        tree_predictions = [predict_one(row, tree) for tree in forest]
        final_predictions.append(majority_class(np.array(tree_predictions)))

    
    return final_predictions