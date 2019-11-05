from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Cleans up data
import preProcess as pp

import linearAlgo
import neuralAlgo as nA


#%matplotlib inline

def createSubmission(arr, dir, id):
    dir = dir + "\\submission\\submission.csv"
    arr = list(arr)
    for i in range(len(arr)):
        arr[i] = [i + 1461, arr[i]]
    df = pd.DataFrame(arr, columns = ["Id", "SalePrice"])
    df.to_csv(dir, index=False)
    print("submission written to file")


def submit(submitB, message, dir):
    dir = dir + "\\submission\\submission.csv"
    command = 'kaggle competitions submit -c house-prices-advanced-regression-techniques -f ' + str(dir) + ' -m "' + str(message) + '"'
    if(submitB):
        os.system(command)
        print("\n Submitted")
    else:
        print("Not Submitted")
        
        

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







##########################################################################################
#####################################DATA PREPROCESSING###################################
#Website for encoding data: https://www.datacamp.com/community/tutorials/categorical-data
# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts2", "")
train_df = load_df(home_dir, "train.csv")
test_df = load_df(home_dir, "test.csv")

all_data = pp.process_data(train_df, test_df)


#Train data at all_data[0], Test at all_data[1], 
#train_y at all_data[2], test_ID at all_data[3]
train_x = all_data[0]
test_x = all_data[1]
train_y = all_data[2]
test_ID = all_data[3]

##########################################################################################







########################################################################################
#####################################Applying ML Algo###################################


predictions = []
titles = []


################## apply Linear Regression: #####################
y_prediction = nA.apply_MLPRegressor(train_x, train_y, test_x)
createSubmission(y_prediction, home_dir, id)


submitD = True
message = "test submission linear Regression"
submit(submitD, message, home_dir)


####################################################################################

################## apply MLP: #####################
y_prediction = nA.apply_MLPRegressor(train_x, train_y, test_x)
createSubmission(y_prediction, home_dir, id)


submitD = True
message = "test submission MultiLayer Perceptron"
submit(submitD, message, home_dir)


####################################################################################

