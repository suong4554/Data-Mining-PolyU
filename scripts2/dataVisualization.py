import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
warnings.filterwarnings('ignore')

home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts2", "")
df_train = pd.read_csv(home_dir + "\\data\\train.csv")
df_test = pd.read_csv(home_dir + "\\data\\test.csv")


#attributes
print(df_train.columns)

#saleprice count, min, max, mean, ...
print(df_train['SalePrice'].describe())

#saleprice skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

##################################################################################################
#plot saleprice
(mu, sigma) = norm.fit(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'] , fit=norm);
plt.ylabel('Frequency')
plt.title('SalePrice distribution before normalization')
plt.legend(['Normal distribution ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.show()

#qqplot saleprice
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.title('Q-Q-Plot SalePrice before normalization')
plt.show()

#plot saleprice normalized
normalized_sale_price = np.log1p(df_train['SalePrice'])
(mu, sigma) = norm.fit(normalized_sale_price)
sns.distplot(normalized_sale_price , fit=norm);
plt.ylabel('Frequency')
plt.title('SalePrice distribution after normalization')
plt.legend(['Normal distribution ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.show()

#qqplot saleprice
fig = plt.figure()
res = stats.probplot(normalized_sale_price, plot=plt)
plt.title('Q-Q-Plot SalePrice after normalization')
plt.show()

##########################################################################################
#normalize grlivarea
#qqplot GrLivArea
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.title('Q-Q-Plot GrLivArea before normalization')
plt.show()

#plot GrLivArea normalized
normalized_grlivarea = np.log1p(df_train['GrLivArea'])

#qqplot GrLivArea
fig = plt.figure()
res = stats.probplot(normalized_grlivarea, plot=plt)
plt.title('Q-Q-Plot GrLivArea after normalization')
plt.show()

####################################################################################################
#scatter plot totalbsmtsf/saleprice
data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));
plt.title('Total square feet of basement area')
plt.show()

#scatter plot grlivarea/saleprice
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
plt.title('Above grade (ground) living area square feet')
plt.show()

# analyze and remove huge outliers: GrLivArea, ...
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

#scatter plot totalbsmtsf/saleprice
data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));
plt.title('Total square feet of basement area')
plt.show()

#scatter plot grlivarea/saleprice
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
plt.title('Above grade (ground) living area square feet')
plt.show()

#box plot overallqual/saleprice
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.title('Rating of overall material and finish of the house')
plt.show()

#box plot yearbuilt/saleprice
data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x='YearBuilt', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
plt.title('Original construction date')
plt.show()


#############################################################################################
# New all encompassing dataset
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

# Dropping the target
all_data.drop(['SalePrice'], axis=1, inplace=True)

# Printing all_data shape
print("all_data size is: {}".format(all_data.shape))

# Getting a missing % count
all_data_missing = (all_data.isnull().sum() / len(all_data)) * 100
all_data_missing = all_data_missing.drop(all_data_missing[all_data_missing == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Percentage':all_data_missing})
missing_data.head(30)

# Visualising missing data
f, ax = plt.subplots(figsize=(10, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Missing Percentage'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Missing data', fontsize=15)
plt.show()

#####################################################################################
# Initiate correlation matrix
corr = df_train.corr()
# Set-up mask
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set-up figure
plt.figure(figsize=(14, 8))
# Title
plt.title('Overall Correlation of House Prices', fontsize=18)
# Correlation matrix
sns.heatmap(corr, mask=mask, annot=False,cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})
plt.show()

#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots()
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix')
plt.show()
