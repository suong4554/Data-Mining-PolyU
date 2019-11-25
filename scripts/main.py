import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import linearAlgo as lA
import neuralAlgo as nA
import stackedAlgo as sA
from preProcess import process_data

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

##########################################################################################
#####################################DATA PREPROCESSING###################################
# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
train_df = load_df(home_dir, "train.csv")
test_df = load_df(home_dir, "test.csv")

[train, test, train_target, test_ID] = process_data(train_df, test_df)

#Split up the dataset for testing and training purposes
train_x, test_x, train_y, test_y = train_test_split(train, train_target, test_size = 0.33, random_state = 5)


########################################################################################
#####################################Applying ML Algo###################################
predictions = []
titles = []

"""
################## apply SVM Regression: #####################
y_prediction = svm.apply_svr(train_x, train_y, test_x)
predictions.append(y_prediction)
titles.append("SVM Regression")
visualize(test_y, y_prediction, "SVM Regression")
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
####################################################################################"""

################## apply Linear Regression #####################
y_prediction = lA.apply_linear_regression(train_x, train_y, train_x)
predictions.append(y_prediction)
titles.append("Linear Regression")
visualize(np.expm1(train_y), y_prediction, "Linear Regression")
####################################################################################

################## apply MLP Regression #####################
y_prediction = nA.apply_MLPRegressor(train_x, train_y, test_x)
predictions.append(y_prediction)
titles.append("MLP Regression")
visualize(np.expm1(test_y), y_prediction, "MLP Regression")
####################################################################################

################## apply Stacked Linear Regression #####################
y_prediction = sA.apply_stacked_regression(train_x, train_y, train_x)
predictions.append(y_prediction)
titles.append("Stacked Linear Regression")
visualize(np.expm1(train_y), y_prediction, "Stacked Linear Regression")
####################################################################################

#for i in range(len(predictions)):
#    visualize(test_y, predictions[i], titles[i])
