from sklearn.neighbors import KNeighborsClassifier


def apply_logistic_regression(train_x, train_y, test_x, size_k):
    # apply k-Nearest-Neighbors Algorithm:
    knn = KNeighborsClassifier(n_neighbors=size_k)
    knn.fit(train_x, train_y)

    # predict the test results:
    y_prediction = knn.predict(test_x)

    # return predictions:
    return y_prediction

