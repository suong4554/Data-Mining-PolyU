from sklearn.neural_network import MLPClassifier


def apply_MLPClassifier(train_x, train_y, test_x):
    # apply Linear Regression:
    mlp = MLPClassifier()
    mlp.fit(train_x, train_y)

    # predict the results:
    y_prediction = mlp.predict(test_x)



    # return predictions:
    return y_prediction
