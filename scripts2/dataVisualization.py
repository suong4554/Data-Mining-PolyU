import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats

home_dir = os.path.dirname(os.path.realpath(__file__)).replace('scripts2', '')
df_train = pd.read_csv(home_dir + '\\data\\train.csv')
df_test = pd.read_csv(home_dir + '\\data\\test.csv')

#attributes
print(df_train.columns)

#saleprice count, min, max, mean, ...
print(df_train['SalePrice'].describe())

#saleprice skewness and kurtosis
print('Skewness: %f' % df_train['SalePrice'].skew())
print('Kurtosis: %f' % df_train['SalePrice'].kurt())

##################################################################################################
#plot saleprice
(mu, sigma) = norm.fit(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'] , fit=norm);
plt.ylabel('Frequency')
plt.title('SalePrice distribution before normalization')
plt.legend(['Normal distribution ($\mu=$ {:.2f}, $\sigma=$ {:.2f} )'.format(mu, sigma)])
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
plt.legend(['Normal distribution ($\mu=$ {:.2f}, $\sigma=$ {:.2f} )'.format(mu, sigma)])
plt.show()

#qqplot saleprice
fig = plt.figure()
res = stats.probplot(normalized_sale_price, plot=plt)
plt.title('Q-Q-Plot SalePrice after normalization')
plt.show()

##########################################################################################
#qqplot GrLivArea
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.title('Q-Q-Plot GrLivArea before normalization')
plt.show()

#qqplot GrLivArea normalized
normalized_grlivarea = np.log1p(df_train['GrLivArea'])
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
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000);
plt.title('Rating of overall material and finish of the house')
plt.show()

#box plot yearbuilt/saleprice
data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
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
print('Data size: {}'.format(all_data.shape))

# Counting missing data
all_data_missing = (all_data.isnull().sum() / len(all_data)) * 100
all_data_missing = all_data_missing.drop(all_data_missing[all_data_missing == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Percentage':all_data_missing})
missing_data.head(30)

# Visualizing missing data
f, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=missing_data.index, y=missing_data['Missing Percentage'])
plt.xticks(rotation='90')
plt.title('Missing data')
plt.show()

#####################################################################################
#correlation matrix
corrmat = df_train.corr()
plt.figure(figsize=(16, 8))
sns.heatmap(corrmat);
plt.title('Correlation of House Prices')
plt.show()
