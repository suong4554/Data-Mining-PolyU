# -*- coding: utf-8 -*-
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

import linearAlgo
import logAlgo
import kncAlgo
import svmAlgo as svm
import neuralAlgo as nA
import gausClassAlgo as gca
import decTreeAlgo as dta
import discrimAnalAlgo as daa
import naiveBayAlgo as nba
import randomForestAlgo as rfa

### STEP1: settings
class Settings(Enum):
    train_path    = 'data/train.csv'
    test_path     = 'data/test.csv'

    def __str__(self):
        return self.value

# returns the data from the Exel table:
def load_df(dir_path, file_name):
    file = dir_path + "data\\" + file_name
    data = pd.read_csv(file)
    return data

def visualize(test_y, pred_y, title):
    plt.scatter(test_y, pred_y)
    plt.xlabel("Prices: $Y_i$")
    plt.ylabel("Predicted prices: $\hat{Y}_i$")
    plt.title(title)
    plt.show()

### STEP2: data processing
def display_outlier(pd, feature):
    fig, ax = plt.subplots()
    ax.scatter(x = pd[feature], y = pd['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel(feature, fontsize=13)
    plt.show()

def display_distrib(pd, feature):
    plt.figure()
    sns.distplot(pd[feature].dropna() , fit=norm);
    (mu, sigma) = norm.fit(pd[feature].dropna())

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    plt.show()

def process_data():
    #print('[data_processing] ', train_path)
    #print('[data_processing] ', test_path)

    home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
    global train
    train = load_df(home_dir, "train.csv")
    global test
    test = load_df(home_dir, "test.csv")

    global y_target
    global train_ID
    global test_ID
    global train_x
    global test_x
    global train_y
    global test_y

    # load data
    #home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
    #train = load_df(home_dir, "train.csv")
    #test = load_df(home_dir, "test.csv")


    # drop ID feature
    print('[data_processing] ', 'The train data size before dropping Id: {} '.format(train.shape))
    print('[data_processing] ', 'The test data size before dropping Id: {} '.format(test.shape))

    train_ID = train['Id']
    test_ID = test['Id']

    train.drop('Id', axis = 1, inplace = True)
    test.drop('Id', axis = 1, inplace = True)

    print('[data_processing] ', 'The train data size after dropping Id: {} '.format(train.shape))
    print('[data_processing] ', 'The test data size after dropping Id: {} '.format(test.shape))

    # analyze and remove huge outliers: GrLivArea, ...
    display_outlier(train, 'GrLivArea')
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
    display_outlier(train, 'GrLivArea')

    # normalize distribution of output (SalePrice)
    display_distrib(train, 'SalePrice')
    train["SalePrice"] = np.log1p(train["SalePrice"])
    y_target = train.SalePrice.values
    display_distrib(train, 'SalePrice')

    # concatenate the train and test data
    ntrain = train.shape[0]
    ntest = test.shape[0]
    train.drop(['SalePrice'], axis=1, inplace=True)
    all_data = pd.concat((train, test)).reset_index(drop=True)
    print('[data_processing] ', 'all_data size is : {}'.format(all_data.shape))

    # fill missing data
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    print('[data_processing] ', missing_data)

    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    all_data["Alley"] = all_data["Alley"].fillna("None")
    all_data["Fence"] = all_data["Fence"].fillna("None")
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

    # add important features more
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] #feature which is the total area of basement, first and second floor areas of each house

    # normalize skewed features
    for feature in all_data:
        if all_data[feature].dtype != "object":
                #display_distrib(all_data, feature)
                all_data[feature] = np.log1p(all_data[feature])
                #display_distrib(all_data, feature)

    # transform numeric features into categorical features
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['OverallQual'] = all_data['OverallQual'].astype(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    # encode categorical features by LabelEncoder or dummies
    # do label encoding for categorical features
    categorical_features = \
    ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
     'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
     'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
     'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallQual',
     'OverallCond', 'YrSold', 'MoSold')
    for c in categorical_features:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))
    print('[data_processing] ', 'Shape all_data: {}'.format(all_data.shape))
    # get dummy categorical features
    all_data = pd.get_dummies(all_data)
    print('[data_processing] ', all_data.shape)

    train = all_data[:ntrain]
    test = all_data[ntrain:]
    print(test.shape, "Test Shape")

    #Split up the dataset for testing and training purposes
    train_x, test_x, train_y, test_y = train_test_split(train, y_target, test_size = 0.33, random_state = 5)


def model():
    predictions = []
    titles = []
    ################## apply Linear Regression: #####################
    y_prediction = linearAlgo.apply_linear_regression(train_x, train_y, test_x)
    predictions.append(y_prediction)
    titles.append("Linear Regression")
    visualize(test_y, y_prediction, "Linear Regression")
    ####################################################################################

    """
    ################## apply SVM Regression: #####################
    y_prediction = svm.apply_svr(train_x, train_y, test_x)
    predictions.append(y_prediction)
    titles.append("SVM Regression")
    visualize(test_y, y_prediction, "SVM Regression")
    ####################################################################################
    """
    ################## apply Neural Network MLP Classifier##################
    y_prediction = nA.apply_MLPRegressor(train_x, train_y, test_x)
    predictions.append(y_prediction)
    titles.append("Neural Network MLP Regressor")
    visualize(test_y, y_prediction, "Neural Network MLP Regressor")
    ####################################################################################

    ################## apply k-Nearest-Neighbors Algorithm:##################
    size_k = 3
    y_prediction = kncAlgo.apply_knn(train_x, train_y, test_x, size_k)
    predictions.append(y_prediction)
    titles.append("k-Nearest-Neighbors Algorithm: k=3")
    visualize(test_y, y_prediction, "k-Nearest-Neighbors Algorithm: k=3")

    size_k = 5
    y_prediction = kncAlgo.apply_knn(train_x, train_y, test_x, size_k)
    predictions.append(y_prediction)
    titles.append("k-Nearest-Neighbors Algorithm: k=4")
    visualize(test_y, y_prediction, "k-Nearest-Neighbors Algorithm: k=5")
    ####################################################################################

    ####create CSV with prediction results#####
    lm = MLPRegressor(1000, 'relu', 'lbfgs')
    print(train)
    print(y_target)
    lm.fit(train, y_target)
    ensemble = np.expm1(lm.predict(test))

    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = ensemble
    sub.to_csv('submission.csv',index=False)


### MAIN
process_data()
model()
