from sklearn.neural_network import MLPRegressor
import numpy as np

def apply_MLPRegressor(train_x, train_y, test_x):
    # apply Linear Regression:
    mlp = MLPRegressor(20000, 'relu', 'lbfgs')
    mlp.fit(train_x, train_y)

    # predict the results:
    # multiple by exponent to get y_prediction
    y_prediction = np.expm1(mlp.predict(test_x))

    # return predictions:
    return y_prediction
