from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


import linearAlgo
import logAlgo
import kncAlgo
import svmAlgo as svm
import neuralAlgo as nA
import gausClassAlgo as gca
import decTreeAlgo as dta
import discrimAnalAlgo as daa
import naiveBayAlgo as nba

#%matplotlib inline



def visualize(test_y, pred_y, title):
    plt.scatter(test_y, pred_y)
    plt.xlabel("Prices: $Y_i$")
    plt.ylabel("Predicted prices: $\hat{Y}_i$")
    plt.title(title)
    plt.show()



def train(train_x, target_y):
    
    #le = preprocessing.LabelEncoder()
    for column in train_x:
        #le.fit(train_x[column])
        #train_x[column] = le.transform(train_x[column])
        train_x[column] = pd.Categorical(train_x[column])
   
    #header = train_x.columns
    #train_x = pd.get_dummies(data=header)
    
    print(train_x)
        
    lm = LinearRegression()
    lm.fit(train_x, target_y)
    return lm



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
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
train_df = load_df(home_dir, "train.csv")

#Encodes the dataframe to ints
train_df = encodeArr(train_df)


# create the training set: (without "target" and "Id" column)
train_x = train_df.drop("SalePrice", axis=1).drop("Id", axis=1)
target_y = train_df["SalePrice"]

#Split up the dataset for testing and training purposes
train_x, test_x, train_y, test_y = train_test_split(train_x, target_y, test_size = 0.33, random_state = 5)




##########################################################################################
#####################################Applying ML Algo###################################


predictions = []
titles = []


################## apply Linear Regression: #####################
y_prediction = linearAlgo.apply_linear_regression(train_x, train_y, test_x)
predictions.append(y_prediction)
titles.append("Linear Regression")
#visualize(test_y, y_prediction, "Linear Regression")

####################################################################################

################## apply Logistic Regression:##################
y_prediction = logAlgo.apply_logistic_regression(train_x, train_y, test_x)
predictions.append(y_prediction)
titles.append("Logistic Regression")
#visualize(test_y, y_prediction, "Logistic Regression")

####################################################################################
####################################################################################

################## apply Neural Network MLP Classifier##################
y_prediction = nA.apply_MLPClassifier(train_x, train_y, test_x)
predictions.append(y_prediction)
titles.append("Neural Network MLP Classifier")
#visualize(test_y, y_prediction, "Neural Network MLP Classifier")
####################################################################################

################## apply Naive Bayes GaussianNB##################
y_prediction = nba.apply_naive(train_x, train_y, test_x)
predictions.append(y_prediction)
titles.append("Naive Bayes GaussianNB")
#visualize(test_y, y_prediction, "Naive Bayes GaussianNB")

#Apply Linear and Quadratic Discrimination Analysis
solvers = ["svd", "lsqr", "eigen"]
#Eigen does not work
#apply linear
y_prediction = daa.apply_linear_disc(train_x, train_y, test_x, "svd")
predictions.append(y_prediction)
titles.append("Linear Discrimination Analysis: svd")
#visualize(test_y, y_prediction, "Linear Discrimination Analysis: svd")

y_prediction = daa.apply_linear_disc(train_x, train_y, test_x, "lsqr")
predictions.append(y_prediction)
titles.append("Linear Discrimination Analysis: lsqr")
#visualize(test_y, y_prediction, "Linear Discrimination Analysis: lsqr")

"""
#apply quadratic
#No solvers for quadratic discrimination
y_prediction = daa.apply_quad_disc(train_x, train_y, test_x)
predictions.append(y_prediction)
titles.append("Quadratic Discrimination Analysis: lsqr")
#visualize(test_y, y_prediction, "Quadratic Discrimination Analysis: lsqr")
"""
####################################################################################


##################apply Gaussian Process Classifier##################
y_prediction = gca.apply_gaus(train_x, train_y, test_x)
predictions.append(y_prediction)
titles.append("Gaussian Process Classifier")
#visualize(test_y, y_prediction, "Gaussian Process Classifier")

####################################################################################

###################apply Decision Tree Classifier##################
y_prediction = dta.apply_tree(train_x, train_y, test_x)
predictions.append(y_prediction)
titles.append("Decision Tree Classifier")
#visualize(test_y, y_prediction, "Decision Tree Classifier")

####################################################################################

################## apply k-Nearest-Neighbors Algorithm:##################
size_k = 3
y_prediction = kncAlgo.apply_logistic_regression(train_x, train_y, test_x, size_k)
predictions.append(y_prediction)
titles.append("k-Nearest-Neighbors Algorithm: k=3")
#visualize(test_y, y_prediction, "k-Nearest-Neighbors Algorithm: k=3")

size_k = 1
y_prediction = kncAlgo.apply_logistic_regression(train_x, train_y, test_x, size_k)
predictions.append(y_prediction)
titles.append("k-Nearest-Neighbors Algorithm: k=1")
#visualize(test_y, y_prediction, "k-Nearest-Neighbors Algorithm: k=1")


####################################################################################


for i in range(len(predictions)):
    visualize(test_y, predictions[i], titles[i])



