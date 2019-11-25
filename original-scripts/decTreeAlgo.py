from sklearn.tree import DecisionTreeRegressor

def apply_tree(train_x, train_y, test_x):
    # apply Linear Regression:
    dct = DecisionTreeRegressor()
    dct.fit(train_x, train_y)

    # predict the results:
    y_prediction = dct.predict(test_x)

    # return predictions:
    return y_prediction
