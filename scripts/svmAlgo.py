from sklearn import svm

#Applies Support Vector Machines
def apply_svc(train_x, train_y, test_x, kernelt, degreet):
    # apply Support Vector Regression
    svc = svm.SVC(kernel=kernelt, degree=degreet, gamma="auto")
    svc.fit(train_x, train_y) 

    # predict the results:
    y_prediction = svc.predict(test_x)



    # return predictions:
    return y_prediction

def apply_svr(train_x, train_y, test_x, kernelt, degreet):
    # apply Support Vector Regression
    svr = svm.SVR(kernel=kernelt, degree=degreet, gamma="auto")
    svr.fit(train_x, train_y) 

    # predict the results:
    y_prediction = svr.predict(test_x)


    # return predictions:
    return y_prediction

