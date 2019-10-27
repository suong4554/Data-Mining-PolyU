import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score



# display result accuracy of learning procedure:
def display_accuracy(y_prediction, test_y, display_message):
    print("Accuracy_Score of " + display_message + " " + str(accuracy_score(y_prediction, test_y.tolist())))



def encodeSingleColumn(train_x, test_x):
    #Selects the Column
    hs_train = train_x[['HouseStyle']].copy()
    hs_test = test_x[['HouseStyle']].copy()
    
    #hs_train = train_x
    #hs_test = test_x
    
    
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    
    #Fill in missing values (nan)
    si = SimpleImputer(strategy='constant', fill_value='MISSING')
    hs_train = si.fit_transform(hs_train)
    hs_test = si.fit_transform(hs_test)
    
    
    #Transforms the data
    hs_train_transformed = ohe.fit_transform(hs_train)
    hs_test_transformed = ohe.transform(hs_test)
    
    print(hs_train_transformed)
    #row0 = hs_train_transformed[0]
    
    
    #fill, transforms
    #inverse_transform([row0])
    #print(ohe.inverse_transform([row0]))

    #ohe = OneHotEncoder(spare=False)
    
    """
    for x, column in train_x.iterrows():
        le.fit(column)
        trainX[x] = le.transform(column)
        print(trainX[x])
    """

#https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
def encodeColumns(train, test):
    from sklearn.compose import ColumnTransformer 
    from sklearn.pipeline import Pipeline
    
    
    kinds = np.array([dt.kind for dt in train.dtypes])
    all_columns = train.columns.values
    is_num = kinds != 'O'
    num_cols = all_columns[is_num]
    cat_cols = all_columns[~is_num]
    
    
    
    cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))
    cat_ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))    
    cat_steps = [cat_si_step, cat_ohe_step]
    cat_pipe = Pipeline(cat_steps)
    cat_transformers = [('cat', cat_pipe, cat_cols)]
    ct = ColumnTransformer(transformers=cat_transformers)
    
    
    from sklearn.preprocessing import StandardScaler
    

    
    num_si_step = ('si', SimpleImputer(strategy='median'))
    num_ss_step = ('ss', StandardScaler())
    num_steps = [num_si_step, num_ss_step]    
    num_pipe = Pipeline(num_steps)
    num_transformers = [('num', num_pipe, num_cols)]
    
    transformers = [('cat', cat_pipe, cat_cols),
                    ('num', num_pipe, num_cols)]
    
    
    ct = ColumnTransformer(transformers=transformers)
    dataWholeTrain = ct.fit_transform(train)
    dataWholeTest = ct.transform(test)
    whole = [dataWholeTrain, dataWholeTest]
    return whole



def train(train_x, target_y):
    
    le = preprocessing.LabelEncoder()
    for x, column in train_x.iterrows():
        #print(column)
        break
        
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
target_y = train_df["SalePrice"]



train_x, test_x, train_y, test_y = train_test_split(train_x, target_y, test_size = 0.33, random_state = 5)
"""
size = len(train_x)

# take "test_amount" examples for testing:
test_amount = int(size*.25)
print(test_amount)
test_x = train_x.tail(test_amount)
test_y = target_y.tail(test_amount)

# the "250-test_amount" are used for training:
train_y = target_y.head(size - test_amount)
train_x = train_x.head(size - test_amount)

"""
"""
#load the testing DataFrame:
train_df = load_df(home_dir, "test.csv")

#create the testing set: (without "id" column)
test_x = train_df.drop("Id", axis=1)
"""

data = encodeColumns(train_x, test_x)
train_x_encoded = data[0]
test_x_encoded = data[1]

print(train_x_encoded.shape)
print(test_x_encoded.shape)
print(train_y.shape)

lm = LinearRegression()
lm.fit(train_x_encoded, train_y)




y_prediction = lm.predict(test_x_encoded)




plt.scatter(test_y, y_prediction)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()



