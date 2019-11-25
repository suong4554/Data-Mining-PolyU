



def cleanupEmptyValues(df):
    #Fills in all NaN values
    fillNones = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'FireplaceQu', 'MasVnrType', 'MSSubClass', 'GarageYrBlt', 'GarageArea', 'GarageCars']
    fillZeros = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrArea']
    fillFrequent = ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'MSZoning', 'SaleType']
    for col in (fillNones):
        df[col] = df[col].fillna("None")
    for col in (fillZeros):
        df[col] = df[col].fillna(0)
    for col in (fillFrequent):
        df[col] =  = df[col].fillna(df[col].mode()[0])
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))  # fill by the median LotFrontage of all neighborhood because they have same lot frontage
    df = df.drop(['Utilities'], axis=1) #For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it
    df["Functional"] = df["Functional"].fillna("Typ") #data description says NA means typical
    return df

def combineDropFeatures(df):
    # add important features more
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] #feature which is the total area of basement, first and second floor areas of each house
    
    
    
    
    
    
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