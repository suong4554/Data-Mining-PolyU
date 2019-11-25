from sklearn.ensemble import RandomForestRegressor

def apply_forest(train_x, train_y, test_x):
    # apply Linear Regression:
    rfr = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=3)
    rfr.fit(train_x, train_y)

    # predict the results:
    y_prediction = rfr.predict(test_x)

    # return predictions:
    return y_prediction
