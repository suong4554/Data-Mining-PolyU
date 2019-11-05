from sklearn.neural_network import MLPRegressor

def apply_MLPRegressor(train_x, train_y, test_x):
    # apply Linear Regression:
    mlp = MLPRegressor(1000, 'relu', 'lbfgs')
    mlp.fit(train_x, train_y)

    # predict the results:
    y_prediction = mlp.predict(test_x)

    # return predictions:
    return y_prediction
