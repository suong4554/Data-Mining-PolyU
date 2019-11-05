from sklearn.linear_model import LinearRegression


def apply_linear_regression(train_x, train_y, test_x):
    # apply Linear Regression:
    lm = LinearRegression()
    lm.fit(train_x, train_y)

    # predict the results:
    y_prediction = lm.predict(test_x)

    # return predictions:
    return y_prediction
