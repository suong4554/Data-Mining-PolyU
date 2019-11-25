from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def apply_linear_disc(train_x, train_y, test_x, solverT):
    # apply Linear Regression:
    lda = LinearDiscriminantAnalysis(solver=solverT)
    lda.fit(train_x, train_y)

    # predict the results:
    y_prediction = lda.predict(test_x)

    # return predictions:
    return y_prediction


#No solvers for quadratic discrimination
def apply_quad_disc(train_x, train_y, test_x):
    # apply Linear Regression:
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(train_x, train_y)

    # predict the results:
    y_prediction = qda.predict(test_x)


    # return predictions:
    return y_prediction
