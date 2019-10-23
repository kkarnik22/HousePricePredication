#loading need libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#Load data for train and test
train = pd.read_csv('Graphs/train.csv')

# displays the first 5 rows of the datset
print(train.head())


#shape of train data
print(train.shape)

#you can also check the data set information using the info() command.
#train.info()
'''
# Here we use log for target variable to make more normal distribution
# we use log function which is in numpy
train['SalePrice'] = np.log1p(train['SalePrice'])

# the following lines from 26-36 are just to plot the data and check that
# after applying log whether it is normally distributed or not

plt.subplots(figsize=(12,9))
sns.distplot(train['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['SalePrice'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
'''
# ************************************* DATA CLEANING ************************************

#Let's check if the data set has any missing values.
train.columns[train.isnull().any()]

#missing value counts in each of these columns
Isnull = train.isnull().sum()/len(train)*100
Isnull = Isnull[Isnull>0]
Isnull.sort_values(inplace=True, ascending=False)


# Visualising missing values
#Convert into dataframe
Isnull = Isnull.to_frame()
Isnull.columns = ['count']
Isnull.index.names = ['Name']
Isnull['Name'] = Isnull.index
print(Isnull)
'''
#plot Missing values
plt.figure(figsize=(13, 5))
sns.set(style='whitegrid')
sns.barplot(x='Name', y='count', data=Isnull)
plt.xticks(rotation = 90)
plt.show()
'''
sns.heatmap(train.isnull()).set_title('Null values heatmap')
plt.show()
# ******************************** Finding coorelation between attributes ************************

#Separate variable into new dataframe from original dataframe which has only numerical values
#there is 38 numerical attribute from 81 attributes
train_corr = train.select_dtypes(include=[np.number])

#Delete Id because that is not need for corralation plot
del train_corr['Id']

print("Find most important features relative to target")
corr = train.corr(method='pearson')
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
print(corr.SalePrice)

# ************************** Inputting missing values ****************************

# PoolQC has missing value ratio is 99%+. So, there is fill by None
train['PoolQC'] = train['PoolQC'].fillna('None')

#Arround 50% missing values attributes have been fill by None
train['MiscFeature'] = train['MiscFeature'].fillna('None')
train['Alley'] = train['Alley'].fillna('None')
train['Fence'] = train['Fence'].fillna('None')
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')

#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# GarageType, GarageFinish, GarageQual and GarageCond these are replacing with None
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train[col] = train[col].fillna('None')

#GarageYrBlt, GarageArea and GarageCars these are replacing with zero
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    train[col] = train[col].fillna(int(0))

#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual these are replacing with None
for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
    train[col] = train[col].fillna('None')


#MasVnrArea : replace with zero
train['MasVnrArea'] = train['MasVnrArea'].fillna(int(0))

#MasVnrType : replace with None
train['MasVnrType'] = train['MasVnrType'].fillna('None')

#There is put mode value
train['Electrical'] = train['Electrical'].fillna(train['Electrical']).mode()[0]

#There is no need of Utilities
train = train.drop(['Utilities'], axis=1)

#Checking there is any null value or not
plt.figure(figsize=(10, 5))
sns.heatmap(train.isnull()).set_title('Heatmap after removing null values')
plt.show()
# Now, there is no any missing values

# ****************************** Encoding str to int *******************************

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature',
        'SaleType', 'SaleCondition', 'Electrical', 'Heating')


from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[c].values))
    train[c] = lbl.transform(list(train[c].values))
print(train.head())
# ******************************** Prepraring data for prediction ***********************

#Take targate variable into y
y = train['SalePrice']

#Delete the saleprice
del train['SalePrice']

#Take their values in X and y
X = train.values
y = y.values

# Split data into train and test formate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# ******************************** Linear Regression *********************************

#Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()

#Fit the model
model.fit(X_train, y_train)

#Prediction
print("\nPredict value " + str(model.predict(X_test)))
print("Real value " + str(y_test))

#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)
