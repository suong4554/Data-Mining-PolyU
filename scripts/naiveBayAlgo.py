from sklearn.naive_bayes import GaussianNB

def apply_naive(train_x, train_y, test_x):
    # apply Linear Regression:
    gnb = GaussianNB()
    gnb.fit(train_x, train_y)

    # predict the results:
    y_prediction = gnb.predict(test_x)

    # return predictions:
    return y_prediction
