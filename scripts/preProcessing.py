import pandas as pd
import numpy as np
import os


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
    
   
def combineLivingArea(df):
    #Combines 1stFloor SquareFeet with 2ndFloor SquareFeet
    #puts it into column floorSF
    first = df["1stFlrSF"]
    second = df["2ndFlrSF"]
    
    total = []
    
    for i, secondVal in enumerate(second):
        firstVal = first[i]
        totalVal = firstVal + secondVal
        total.append(totalVal)
    
    temp = pd.DataFrame({"floorSF": total}) 
        
def visualize(test_y, pred_y, title, column):
    plt.scatter(test_y, pred_y)
    plt.xlabel(column)
    plt.ylabel("Prices: $Y_i$")
    plt.title(title)
    plt.show() 

   
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
train_df = load_df(home_dir, "train.csv")

combineLivingArea(train_df)

#1stFlrSF
#2ndFlrSF
