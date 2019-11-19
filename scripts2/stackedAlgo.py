import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def rmse_cv(model, train_x, train_y):
    rmse = np.sqrt(-cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)

def apply_stacked_regression(train_x, train_y, test_x):
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 30]
    cv_ridge = [rmse_cv(Ridge(alpha = a), train_x, train_y).mean() for a in alphas]
    cv_ridge = pd.Series(cv_ridge, index = alphas)
    print(cv_ridge.min)
    cv_ridge.plot(title = "Finding alpha for ridge regression")
    plt.xlabel("Alpha")
    plt.ylabel("Rmse")
    plt.show()

    alphas = [0.01, 0.005, 0.001, 0.0006, 0.0005, 0.0004, 0.0003, 0.0001]
    cv_lasso = [rmse_cv(Lasso(alpha = a), train_x, train_y).mean() for a in alphas]
    cv_lasso = pd.Series(cv_lasso, index = alphas)
    print(cv_lasso.min)
    cv_lasso.plot(title = "Finding alpha for lasso regression")
    plt.xlabel("Alpha")
    plt.ylabel("Rmse")
    plt.show()

    alphas = [0.01, 0.005, 0.001, 0.0006, 0.0005, 0.0004, 0.0003, 0.0001]
    cv_elastic = [rmse_cv(ElasticNet(alpha = a), train_x, train_y).mean() for a in alphas]
    cv_elastic = pd.Series(cv_elastic, index = alphas)
    print(cv_elastic.min)
    cv_elastic.plot(title = "Finding alpha for elastic net regression")
    plt.xlabel("Alpha")
    plt.ylabel("Rmse")
    plt.show()

    alphas = [1, 0.01, 0.0006, 0.0001]
    cv_krr = [rmse_cv(KernelRidge(alpha = alpha, kernel='polynomial'), train_x, train_y).mean() for alpha in alphas]
    cv_krr = pd.Series(cv_krr, index = alphas)
    print(cv_krr.min)
    cv_krr.plot(title = "Finding alpha for kernel ridge regression")
    plt.xlabel("Alpha")
    plt.ylabel("Rmse")
    plt.show()

    #Ridge regression: adds an additional term to the cost function,sums the squares of coefficient values (L-2 norm) and multiplies it by a constant alpha
    model_ridge = Ridge(alpha = 10)
    #Lasso regression: adds an additional term to the cost function, sums the coefficient values (L-1 norm) and multiplies it by a constant alpha
    model_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0004))
    #Elastic net regression: includes both L-1 and L-2 norm regularization terms, benefits of Lasso and Ridge regression
    model_elastic = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0006))
    #Kernel Ridge regression: combines Ridge Regression (L2-norm regularization) with kernel trick
    model_krr = make_pipeline(RobustScaler(), KernelRidge(alpha=0.01, kernel='polynomial'))

    # Fit and predict all models
    model_lasso.fit(train_x, train_y)
    lasso_pred = np.expm1(model_lasso.predict(test_x))
    model_elastic.fit(train_x, train_y)
    elastic_pred = np.expm1(model_elastic.predict(test_x))
    model_ridge.fit(train_x, train_y)
    ridge_pred = np.expm1(model_ridge.predict(test_x))
    model_krr.fit(train_x, train_y)
    krr_pred = np.expm1(model_krr.predict(test_x))

    # Create stacked model
    y_prediction = (lasso_pred + elastic_pred + ridge_pred + krr_pred) / 4
    return y_prediction
