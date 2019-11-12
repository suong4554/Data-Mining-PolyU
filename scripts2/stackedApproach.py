from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from preProcess import process_data

from sklearn.linear_model import Lasso, Ridge, RidgeCV, ElasticNet, LinearRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from catboost import Pool, CatBoostRegressor, cv

def visualize(test_y, pred_y, title):
    plt.scatter(test_y, pred_y)
    plt.xlabel("Prices: $Y_i$")
    plt.ylabel("Predicted prices: $\hat{Y}_i$")
    plt.title(title)
    plt.show()


# returns the data from the Exel table:
def load_df(dir_path, file_name):
    file = dir_path + "\\data\\" + file_name
    data = pd.read_csv(file)
    return data

def encodeArr(train_df):
    for i in train_df:
        #Casting type to category for efficiency
        train_df[i] = train_df[i].fillna(train_df[i]).astype('category')
        #Built in python to convert each value in a column to a number
        train_df[i] = train_df[i].cat.codes
    return train_df

##########################################################################################
#####################################DATA PREPROCESSING###################################
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts2", "")
train_df = load_df(home_dir, "train.csv")
test_df = load_df(home_dir, "test.csv")

[train, test, train_target, test_ID] = process_data(train_df, test_df)


#################################### New Approach starting here ###########################
#########################################################################################

#Ridge regression: adds an additional term to the cost function,sums the squares of coefficient values (L-2 norm) and multiplies it by a constant alpha
model_ridge = Ridge(alpha = 5)

#Lasso regression: adds an additional term to the cost function, sums the coefficient values (L-1 norm) and multiplies it by a constant alpha
model_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005))

#Elastic net regression: includes both L-1 and L-2 norm regularization terms, benefits of both Lasso and Ridge regression
model_elastic = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0005))

#Kernel Ridge regression: combines Ridge Regression (L2-norm regularization) with kernel trick.
model_krr = make_pipeline(RobustScaler(), KernelRidge(alpha=6, kernel='polynomial', degree=2.65, coef0=6.9))


# Initiating Gradient Boosting Regressor
model_gbr = GradientBoostingRegressor()

# Initiating XGBRegressor
model_xgb = xgb.XGBRegressor()

# Initiating LGBMRegressor model
model_lgb = lgb.LGBMRegressor()

# Fit and predict all models
model_lasso.fit(train, train_target)
lasso_pred = np.expm1(model_lasso.predict(test))

model_elastic.fit(train, train_target)
elastic_pred = np.expm1(model_elastic.predict(test))

model_ridge.fit(train, train_target)
ridge_pred = np.expm1(model_ridge.predict(test))

model_krr.fit(train, train_target)
krr_pred = np.expm1(model_krr.predict(test))

model_xgb.fit(train, train_target)
xgb_pred = np.expm1(model_xgb.predict(test))

model_gbr.fit(train, train_target)
gbr_pred = np.expm1(model_gbr.predict(test))

model_lgb.fit(train, train_target)
lgb_pred = np.expm1(model_lgb.predict(test))

# Create stacked model
stacked = (lasso_pred + elastic_pred + ridge_pred + krr_pred + xgb_pred + lgb_pred + gbr_pred) / 7
#stacked = (lasso_pred + elastic_pred + ridge_pred + krr_pred) / 4

# Setting up competition submission
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = stacked
sub.to_csv('submission.csv',index=False)
