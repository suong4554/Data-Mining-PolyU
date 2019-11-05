import pandas as pd


    
    

def dropColumn(df, dropArr):
    for item in dropArr:
        df = df.drop(item, axis = 1)
        
    return df



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