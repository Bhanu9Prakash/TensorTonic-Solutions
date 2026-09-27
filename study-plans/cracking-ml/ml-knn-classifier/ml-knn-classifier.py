import numpy as np

def knn_classifier(X_train: list, y_train: list, X_test: list, k: int) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    predictions = []
    for test_point in X_test:
        distances = []
        for x in X_train:
            distance = np.sqrt(np.sum((x - test_point)**2))
            distances.append(distance)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        prediction = np.argmax(np.bincount(nearest_labels))
        predictions.append(prediction)


    return predictions
