from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import numpy as np

def apply_stacked_regression(train_x, train_y, test_x):
    #Ridge regression: adds an additional term to the cost function,sums the squares of coefficient values (L-2 norm) and multiplies it by a constant alpha
    model_ridge = Ridge(alpha = 5)
    #Lasso regression: adds an additional term to the cost function, sums the coefficient values (L-1 norm) and multiplies it by a constant alpha
    model_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005))
    #Elastic net regression: includes both L-1 and L-2 norm regularization terms, benefits of both Lasso and Ridge regression
    model_elastic = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0005))
    #Kernel Ridge regression: combines Ridge Regression (L2-norm regularization) with kernel trick.
    model_krr = make_pipeline(RobustScaler(), KernelRidge(alpha=6, kernel='polynomial', degree=2.65, coef0=6.9))

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
