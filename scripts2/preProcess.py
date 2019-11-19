import numpy as np
import pandas as pd
import featureCombine as fc
from sklearn.preprocessing import LabelEncoder

def cleanupEmptyValues(df):
    #Fills in all NaN values
    fillNones = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType', 'MSSubClass', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    fillZeros = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',  'GarageYrBlt', 'GarageArea', 'GarageCars']
    fillFrequent = ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'MSZoning', 'SaleType']
    fillNones = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType', 'MSSubClass', 'GarageYrBlt', 'GarageArea', 'GarageCars']
    fillZeros = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrArea']
    fillFrequent = ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'MSZoning', 'SaleType']
    for col in (fillNones):
        df[col] = df[col].fillna("None")
    for col in (fillZeros):
        df[col] = df[col].fillna(0)
    for col in (fillFrequent):
        df[col] = df[col].fillna(df[col].mode()[0])
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))  # fill by the median LotFrontage of all neighborhood because they have same lot frontage
    df = df.drop(['Utilities'], axis=1) #For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it
    df["Functional"] = df["Functional"].fillna("Typ") #data description says NA means typical
    return df


def combineDropFeatures(df):
    # add important features more
    #df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] #feature which is the total area of basement, first and second floor areas of each house

    df = fc.combineLivingArea(df)
    df = fc.combineBaths(df)
    df = fc.mergeYearBuilt(df)
    #df = combineUtilities(df)
    #df = porchTypes(df)

    '''
    combine = [
        ["WoodDeckSF", "OpenPorchSF", "OutdoorSpace"],
        ["BsmtFinType1", "BsmtFinType2", "BsmtFin"],
        ["ExterQual", "ExterCond", "ExterEval"],
        ["BsmtQual", "BsmtCond", "BsmtEval"],
        ["Condition1", "Condition2", "OverallProx"],
        ["Exterior1st", "Exterior2nd", "Exterior"]
        ]
    '''
    #df = calculateValue(df, combine)

    return df


def normalizeFeatures(df):
    # normalize skewed features
    for feature in df:
        if df[feature].dtype != "object":
                #display_distrib(df, feature)
                df[feature] = np.log1p(df[feature])
                #display_distrib(df, feature)

    return df


def numericToCategory(df):
    # transform numeric features into categorical features
    df['MSSubClass'] = df['MSSubClass'].apply(str)
    df['OverallQual'] = df['OverallQual'].astype(str)
    df['OverallCond'] = df['OverallCond'].astype(str)
    df['YrSold'] = df['YrSold'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)

    # encode categorical features by LabelEncoder or dummies
    # do label encoding for categorical features
    categorical_features = \
        ('FireplaceQu',
        'BsmtQual',
        'BsmtCond',
        'GarageQual',
        'GarageCond',
        'ExterQual',
        'ExterCond',
        'HeatingQC',
        'PoolQC',
        'KitchenQual',
        'BsmtFinType1',
        'BsmtFinType2',
        'Functional',
        'Fence',
        'BsmtExposure',
        'GarageFinish',
        'LandSlope',
        'LotShape',
        'PavedDrive',
        'Street',
        'Alley',
        'CentralAir',
        'MSSubClass',
        'OverallQual',
        'OverallCond',
        'YrSold',
        'MoSold')
    for c in categorical_features:
        lbl = LabelEncoder()
        lbl.fit(list(df[c].values))
        df[c] = lbl.transform(list(df[c].values))

    # get dummy categorical features
    df = pd.get_dummies(df)

    return df


def process_data(train, test):

    train_ID = train['Id']
    test_ID = test['Id']
    train.drop('Id', axis = 1, inplace = True)
    test.drop('Id', axis = 1, inplace = True)

    # analyze and remove huge outliers: GrLivArea, ...
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

    # normalize distribution of output (SalePrice)
    train["SalePrice"] = np.log1p(train["SalePrice"])
    train_target = train.SalePrice.values

    # concatenate the train and test data
    trainShape = train.shape[0]
    train.drop(['SalePrice'], axis=1, inplace=True)
    df = pd.concat((train, test)).reset_index(drop=True)

    df = cleanupEmptyValues(df)
    df = combineDropFeatures(df)
    df = normalizeFeatures(df)
    df = numericToCategory(df)

    train = df[:trainShape]
    test = df[trainShape:]

    print(test.shape)
    #Train data at all_data[0], Test at all_data[1],
    #train_target at all_data[2], test_ID at all_data[3]
    all_data = [train, test, train_target, test_ID]

    return all_data
