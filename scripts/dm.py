from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#%matplotlib inline



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




##########################################################################################
#####################################DATA PREPROCESSING###################################
#Website for encoding data: https://www.datacamp.com/community/tutorials/categorical-data
# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
train_df = load_df(home_dir, "train.csv")

#train_df = pd.read_csv("train.csv")

# performs a mode imputation for those null values
#train_df = train_df.fillna(train_df['Alley'].value_counts().index[0])

for i in train_df:
    #Casting type to category for efficiency
    train_df[i] = train_df[i].fillna(train_df[i]).astype('category')
    #Built in python to convert each value in a column to a number
    train_df[i] = train_df[i].cat.codes
    
    


# create the training set: (without "target" and "Id" column)
train_x = train_df.drop("SalePrice", axis=1).drop("Id", axis=1)
target_y = train_df["SalePrice"]


train_x, test_x, train_y, test_y = train_test_split(train_x, target_y, test_size = 0.33, random_state = 5)


"""
# take "test_amount" examples for testing:
test_amount = 75
test_x = train_x.tail(test_amount)
test_y = target_y.tail(test_amount)
# the "250-test_amount" are used for training:
train_y = target_y.head(250 - test_amount)
train_x = train_x.head(250 - test_amount)

"""


# create the training set: (without "SalePrice" and "Id" column)
train_x = train_df.drop("SalePrice", axis=1).drop("Id", axis=1)
train_y = train_df["SalePrice"]

print(train_x)


lm = train(train_x, train_y)
#lm.fit(train_x, train_y)


pred_y = lm.predict(test_x)



plt.scatter(test_y, pred_y)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()