import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import linearAlgo
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
import neuralAlgo as nA


from sklearn.model_selection import train_test_split



def submit(submitB, message, dir):
    dir = dir + "\\submission\\submission.csv"
    command = 'kaggle competitions submit -c house-prices-advanced-regression-techniques -f ' + str(dir) + ' -m "' + str(message) + '"'
    if(submitB):
        os.system(command)
        print("\n Submitted")
    else:
        print("Not Submitted")



def load_df(dir_path, file_name):
    file = dir_path + "\\data\\" + file_name
    data = pd.read_csv(file)
    return data

def createSubmission(arr, dir, id):
    dir = dir + "\\submission\\submission.csv"
    arr = list(arr)
    for i in range(len(arr)):
        arr[i] = [i + 1461, arr[i]]
    df = pd.DataFrame(arr, columns = ["Id", "SalePrice"])
    df.to_csv(dir, index=False)
    print("submission written to file")




def dropColumn(df, dropArr):
    for item in dropArr:
        df = df.drop(item, axis = 1)
        
    return df


def encodeArr(train_df):
    for i in train_df:
        #Casting type to category for efficiency
        train_df[i] = train_df[i].fillna(train_df[i]).astype('category')
        #Built in python to convert each value in a column to a number
        train_df[i] = train_df[i].cat.codes
    return train_df
    
   


def calculateValue(df, arrQC):
    #Merges Quality and Condition
    
    for subset in arrQC:
        columns = [subset[0], subset[1]]
        
        cond1 = df[subset[0]]
        cond2 = df[subset[1]]

        total = []
        for i, cond1Val in enumerate(cond1):
            cond2Val = cond2[i]
            totalVal = cond2Val + cond1Val
            total.append(totalVal)
        
        
        df[subset[2]] = total
        df = dropColumn(df, columns)
        
        
    return df



def combineBaths(df):
    #Combines all bathrooms
    #puts it into column totalBaths
    
    baths = ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"]
    
    #hb = half bath, fb = full bath
    bhb = df["BsmtFullBath"]
    bfb = df["BsmtHalfBath"]
    fb = df["FullBath"]
    hb = df["HalfBath"]

    total = []
    for i, bhbVal in enumerate(bhb):
        bfbVal = bfb[i]
        fbVal = fb[i]
        hbVal = hb[i]
        totalVal = bhbVal*.5 + bfbVal + fbVal + hbVal*.5
        total.append(totalVal)
    
    
    df["totalBaths"] = total
    df = dropColumn(df, baths)
    return df




def porchTypes(df):
    #Combines EnclosedPorch, 3SsnPorch and ScreenPorch (once both are converted to ints)
    #puts it into column PorchTypes
    
    PorchTypes = ["EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    
    Enclosed = df["EnclosedPorch"]
    TriSsn = df["3SsnPorch"]
    Screen = df["ScreenPorch"]

    total = []
    for i, EnclosedVal in enumerate(Enclosed):
        TriSsnVal = TriSsn[i]
        ScreenVal = Screen[i]
        totalVal = TriSsnVal + ScreenVal + EnclosedVal
        total.append(totalVal)
    
    
    df["PorchTypes"] = total
    df = dropColumn(df, PorchTypes)
    return df 
    


def mergeYearBuilt(df):
    #Combines YearBuilt and YearRemodAdd (once both are converted to ints)
    #puts it into column YearRennovation
    years = ["YearBuilt", "YearRemodAdd"]
    
    cond1 = df["YearBuilt"]
    cond2 = df["YearRemodAdd"]
    total = []
    for i, cond1Val in enumerate(cond1):
        cond2Val = cond2[i]
        recentVal = 0
        if(cond1Val > cond2Val):
            recentVal = cond1Val
        else:
            recentVal = cond2Val
        total.append(recentVal)
    
    
    df["YearRennovation"] = total
    df = dropColumn(df, years)
    return df
    
      
def combineLivingArea(df):
    #Combines 1stFloor SquareFeet with 2ndFloor SquareFeet
    #puts it into column totalSF
    
    livingSF = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]
    
    ground = df["TotalBsmtSF"]
    first = df["1stFlrSF"]
    second = df["2ndFlrSF"]

    total = []
    for i, secondVal in enumerate(second):
        firstVal = first[i]
        groundVal = ground[i]
        totalVal = groundVal + firstVal + secondVal
        total.append(totalVal)
    
    
    df["totalSF"] = total
    df = dropColumn(df, livingSF)
    return df
 
def combineUtilities(df):
    #Combines Heating, CentralAir and Electrical
    #puts it into column utilities
    
    livingSF = ["Heating", "CentralAir", "Electrical"]
    
    heating = df["Heating"].fillna(df["Heating"]).astype('category').cat.codes
    ac = df["CentralAir"].fillna(df["Heating"]).astype('category').cat.codes
    electrical = df["Electrical"].fillna(df["Heating"]).astype('category').cat.codes

    total = []
    for i, electricalVal in enumerate(electrical):
        heatingVal = heating[i]
        acVal = ac[i]
        utilities = int(heatingVal) + int(acVal) + int(electricalVal)
        total.append(utilities)
    
    
    df["utilities"] = total
    df = dropColumn(df, livingSF)
    return df
    
def visualize(test_y, pred_y, title, column):
    plt.scatter(test_y, pred_y)
    plt.xlabel(column)
    plt.ylabel("Prices: $Y_i$")
    plt.title(title)
    plt.show() 


def encodeCategories(df):
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
        lbl.fit(list(df[c].values))
        df[c] = lbl.transform(list(df[c].values))
        
    return df


def fillMissingValues(df):
    df["PoolQC"] = df["PoolQC"].fillna("None") #data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
    df["MiscFeature"] = df["MiscFeature"].fillna("None") #data description says NA means "no misc feature"
    df["Alley"] = df["Alley"].fillna("None") #data description says NA means "no alley access"
    df["Fence"] = df["Fence"].fillna("None") #NA means "no fence"
    df["FireplaceQu"] = df["FireplaceQu"].fillna("None") #NA means "no fireplace"
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))  # fill by the median LotFrontage of all neighborhood because they have same lot frontage
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        df[col] = df[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        df[col] = df[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        df[col] = df[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df[col] = df[col].fillna('None') #NaN means that there is no basement
    df["MasVnrType"] = df["MasVnrType"].fillna("None") #NA means no masonry veneer
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0) #NA means no masonry veneer
    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
    df = df.drop(['Utilities'], axis=1) #For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it
    df["Functional"] = df["Functional"].fillna("Typ") #data description says NA means typical
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0]) #It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0]) #Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0]) #Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0]) #Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0]) #Fill in again with most frequent which is "WD"
    df['MSSubClass'] = df['MSSubClass'].fillna("None") #Na most likely means No building class. We can replace missing values with None
    return df



def preProcess(df):

    df = fillMissingValues(df)
    df = encodeCategories(df)
    df = combineLivingArea(df)
    df = combineBaths(df)
    df = mergeYearBuilt(df)
    df = combineUtilities(df)
    df = porchTypes(df)
    
    
    combine = [
        ["WoodDeckSF", "OpenPorchSF", "OutdoorSpace"],
        ["BsmtFinType1", "BsmtFinType2", "BsmtFin"],
        ["ExterQual", "ExterCond", "ExterEval"],
        ["BsmtQual", "BsmtCond", "BsmtEval"],
        ["Condition1", "Condition2", "OverallProx"],
        ["Exterior1st", "Exterior2nd", "Exterior"]
        ]
    
    df = calculateValue(df, combine)
    
    return df
   
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
train_df = load_df(home_dir, "train.csv")



#filters out training data
print(train_df.shape)
train_df = preProcess(train_df)
print(train_df.shape)


#Encodes the dataframe to ints
train_df = encodeArr(train_df)





home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
test_df = load_df(home_dir, "test.csv")



#filters out training data
print(test_df.shape)
test_df = preProcess(test_df)
print(test_df.shape)


#Encodes the dataframe to ints
test_df = encodeArr(test_df)
test_x = test_df.drop("Id", axis=1)

#####################Creates Test Set################################3

# create the training set: (without "target" and "Id" column)
train_x = train_df.drop("SalePrice", axis=1).drop("Id", axis=1)
train_y = train_df["SalePrice"]


#Split up the dataset for testing and training purposes
#train_x, test_x, train_y, test_y = train_test_split(train_x, target_y, test_size = 0.33, random_state = 5)




################## apply Linear Regression: #####################
y_prediction = linearAlgo.apply_linear_regression(train_x, train_y, test_x)
#visualize(test_y, y_prediction, "Linear Regression", "Predicted Prices")


################## apply Linear Regression: #####################
y_prediction = nA.apply_MLPRegressor(train_x, train_y, test_x)
createSubmission(y_prediction, home_dir, id)


submitD = True
message = "test submission linear Regression"
submit(submitD, message, home_dir)



################## apply Neural Network MLP Classifier##################
y_prediction = nA.apply_MLPRegressor(train_x, train_y, test_x)
#visualize(test_y, y_prediction, "Neural Network MLP Regressor", "Predicted Prices")
