from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF

def apply_gaus(train_x, train_y, test_x):
    # apply Linear Regression:
    gpc = GaussianProcessClassifier()
    gpc.fit(train_x, train_y)

    # predict the results:
    y_prediction = gpc.predict(test_x)


    # return predictions:
    return y_prediction.tolist()

