from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


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
# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
train_df = load_df(home_dir, "train.csv")

# create the training set: (without "SalePrice" and "Id" column)
train_x = train_df.drop("SalePrice", axis=1).drop("Id", axis=1)
train_y = train_df["SalePrice"]

print(train_x)


lm = train(train_x, train_y)
#lm.fit(train_x, train_y)


Y_pred = lm.predict(X_test)



plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")