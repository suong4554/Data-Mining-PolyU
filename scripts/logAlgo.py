from sklearn.linear_model import LogisticRegression



def apply_logistic_regression(train_x, train_y, test_x):
    # apply Logistic Regression:
    lr = LogisticRegression(
        
        C=1.0, 
        class_weight=None, 
        dual=False, 
        fit_intercept=True,
        intercept_scaling=1, 
        max_iter=100, 
        multi_class='warn',
        n_jobs=None, 
        penalty='l1', 
        random_state=None, 
        #solver='lbfgs',
        tol=0.0001, 
        verbose=0, 
        warm_start=False)
    lr.fit(train_x, train_y)

    # predict the test results:
    y_prediction = lr.predict(test_x)

    # return predictions:
    return y_prediction.tolist()

