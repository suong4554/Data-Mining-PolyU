from sklearn.linear_model import LinearRegression
import numpy as np

def apply_linear_regression(train_x, train_y, test_x):
    # apply Linear Regression
    lr = LinearRegression()
    lr.fit(train_x, train_y)

    # predict the results
    # multiple by exponent to get y_prediction
    y_prediction = np.expm1(lr.predict(test_x))

    # return predictions
    return y_prediction
