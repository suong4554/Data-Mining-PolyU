from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def encodeArr(train_df):
    for i in train_df:
        #Casting type to category for efficiency
        train_df[i] = train_df[i].fillna(train_df[i]).astype('category')
        #Built in python to convert each value in a column to a number
        train_df[i] = train_df[i].cat.codes
    return train_df

def visualize(test_y, pred_y, title, column):
    plt.scatter(test_y, pred_y)
    plt.xlabel(column)
    plt.ylabel("Prices: $Y_i$")
    plt.title(title)
    plt.show()
    
  
# returns the data from the Exel table:
def load_df(dir_path, file_name):
    file = dir_path + "\\data\\" + file_name
    data = pd.read_csv(file)
    return data  
    
##########################################################################################
#####################################DATA PREPROCESSING###################################
#Website for encoding data: https://www.datacamp.com/community/tutorials/categorical-data
# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "").replace("dataVisualization", "")
train_df = load_df(home_dir, "train.csv")

#Encodes the dataframe to ints
train_df = encodeArr(train_df)


# create the training set: (without "target" and "Id" column)
train_x = train_df.drop("SalePrice", axis=1).drop("Id", axis=1)
target_y = train_df["SalePrice"]
##########################################################################################


for i, column in enumerate(train_x.columns):
    print(column)
    data = train_x[column]
    title = column
    visualize(data, target_y, title,column)

