from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


import linearAlgo
import neuralAlgo as nA
from sklearn.neural_network import MLPRegressor
"""import logAlgo
import kncAlgo
import svmAlgo as svm
import neuralAlgo as nA
import gausClassAlgo as gca
import decTreeAlgo as dta
import discrimAnalAlgo as daa
import naiveBayAlgo as nba
import randomForestAlgo as rfa"""

from preProcess import process_data

#%matplotlib inline

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
#Website for encoding data: https://www.datacamp.com/community/tutorials/categorical-data
# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts2", "")
train_df = load_df(home_dir, "train.csv")
test_df = load_df(home_dir, "test.csv")

#Encodes the dataframe to ints
#train_df = encodeArr(train_df)


# create the training set: (without "target" and "Id" column)
#train_x = train_df.drop("SalePrice", axis=1).drop("Id", axis=1)[["MSSubClass", "MSZoning", "LotArea", "Neighborhood", "Condition1", "BldgType", "HouseStyle", "OverallQual", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF", "Electrical", "1stFlrSF", "2ndFlrSF", "GrLivArea", "KitchenQual", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "SaleType"]]

[train, test, train_target, test_ID] = process_data(train_df, test_df)

#Split up the dataset for testing and training purposes
train_x, test_x, train_y, test_y = train_test_split(train, train_target, test_size = 0.33, random_state = 5)



########################################################################################
#####################################Applying ML Algo###################################


predictions = []
titles = []


################## apply Linear Regression: #####################
"""y_prediction = linearAlgo.apply_linear_regression(train_x, train_y, test_x)
predictions.append(y_prediction)
titles.append("Linear Regression")
visualize(np.expm1(test_y), np.expm1(y_prediction), "Linear Regression")"""
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
"""y_prediction = nA.apply_MLPRegressor(train_x, train_y, test_x)
predictions.append(y_prediction)
titles.append("Neural Network MLP Regressor")
visualize(np.expm1(test_y), np.expm1(y_prediction), "Neural Network MLP Regressor")"""
####################################################################################
"""

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
"""

lm = MLPRegressor(25000, 'relu', 'lbfgs')
lm.fit(train, train_target)
ensemble = np.expm1(lm.predict(test))

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)

#for i in range(len(predictions)):
#    visualize(test_y, predictions[i], titles[i])
