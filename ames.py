import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
print("Downloading Ames Housing dataset... this might take a moment")
#using ID 42165 which is standard Ames Housing dataset on OpenML
housing = fetch_openml(data_id=42165, as_frame=True, parser='auto')
ames = housing.frame
print(f"Dataset Loaded: {ames.shape[0]} rows, {ames.shape[1]} columns") 

#looking at the target variable - SalePrice
plt.figure(figsize=(10,6))
sns.histplot(ames['SalePrice'], kde=True, color='blue')
plt.title('Distribution of Sale Prices (Original)')
plt.xlabel('Price ($)')
plt.show() 

#calculating Skewness
print(f"Skewness: {ames['SalePrice'].skew():.2f}") 
#applying log transformation
#using log1p (log(1+x)) to avoid errors if price is ever 0
ames['SalePrice_Log'] = np.log1p(ames['SalePrice'])
#comparing before and after
fig, ax = plt.subplots(1, 2, figsize=(14, 5)) 

#creating the Log Transformed column first
ames['SalePrice_Log'] = np.log1p(ames['SalePrice'])

#creating the blank canvases
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

#plotting first plot (ax[0])
sns.histplot(ames['SalePrice'], kde=True, ax=ax[0], color='red')
ax[0].set_title(f"Original Price (Skew: {ames['SalePrice'].skew():.2f})")

#plotting second plot (ax[1])
sns.histplot(ames['SalePrice_Log'], kde=True, ax=ax[1], color='green')
ax[1].set_title(f"Log Price (Skew: {ames['SalePrice_Log'].skew():.2f})")
plt.show()

#handling missing data
#calculating missing % per column
missing = ames.isnull().sum()
missing = missing[missing > 0]
missing_percent = (missing / len(ames)) * 100

#sorting cleaned data, adding visuals
plt.figure(figsize=(12, 6))
sns.barplot(x=missing_percent.index[:20], y=missing_percent.values[:20], palette='viridis')
plt.xticks(rotation=90)
plt.title("Top 20 Columns with Missing Values (%)")
plt.ylabel("% Missing")
plt.show() 

#categorical Columns where NaN means "Does not exist"
#filling these with the string "None"
cols_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                  'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                  'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                  'MasVnrType']

for col in cols_fill_none:
    ames[col] = ames[col].fillna("None")

#numerical Columns where NaN means "0" (No garage = 0 cars)
cols_fill_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                  'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                  'MasVnrArea']

for col in cols_fill_zero:
    ames[col] = ames[col].fillna(0)

#real missing values (LotFrontage)
#for street frontage, filling with the median of the neighborhood (or just global median.. idk)
ames['LotFrontage'] = ames['LotFrontage'].fillna(ames['LotFrontage'].median())

#filling remaining tiny gaps (like Electrical) with the most common value (Mode)
for col in ames.columns:
    if ames[col].isnull().sum() > 0:
        ames[col] = ames[col].fillna(ames[col].mode()[0])

#FEATURE ENGINEERING
#creating a "Total Square Footage" column
#this combines Basement + 1st Floor + 2nd Floor
ames['TotalSF'] = ames['TotalBsmtSF'] + ames['1stFlrSF'] + ames['2ndFlrSF']

#crosschecks
print(f"Remaining Missing Values: {ames.isnull().sum().sum()}")

#encoding and splitting
from sklearn.model_selection import train_test_split
#separating target (y) and features (x)
#using LOG price as target
y = ames['SalePrice_Log']

#dropping useless things from features
x = ames.drop(['SalePrice', 'SalePrice_Log', 'Id'], axis=1) 

#one-hot encoding
#converting all text columns into binary columns
x = pd.get_dummies(x, drop_first=True)
print(f"New Feature Count: {x.shape[1]}")

#splitting into train ans test
#using 80/20 split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Data Successfully Split!") 

#regularization model using ridge regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
#training model
#using alpha=10 as its the strength of the regularization
model = Ridge(alpha=10)
model.fit(x_train, y_train) 

#prediction
y_pred_log = model.predict(x_test)
#evaluation (RMSE)
#calculating errors
rmse = np.sqrt(mean_squared_error(y_test, y_pred_log))
print(f"Log RMSE: {rmse:.4}")
#converting back to $
#using first 5 predcitons
real_predictions = np.expm1(y_pred_log[:5])
real_actuals = np.expm1(y_test[:5].values)

print("\n--- Real Dollar Check (First % Houses) ---")
for pred, act in zip(real_predictions, real_actuals):
    print(f"Predicted: ${pred:,.0} | Actual: ${act:,.0f} | Diff: ${pred - act:,.0f}")

#generating first final report visuals
#converting back from Log scale to real $ for plot
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test_real, y=y_pred_real, alpha=0.6, color='blue') 

#visuals for prediction
max_val = max(y_test_real.max(), y_pred_real.max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2)
plt.title(f'Actual vs. Predicted Prices (Log RMSE: {0.139})')
plt.xlabel('Actual Price($)')
plt.ylabel('Predicted Price ($)')
plt.grid(True, alpha=0.3)
plt.show() 

#generating second final report visuals
#creating dataframe of coefficient
coefs = pd.Series(model.coef_, index=x.columns)
#sorting into top 10 positives and negatives
imp_coefs = pd.concat([coefs.sort_values().head(10),
                       coefs.sort_values().tail(10)])
#plots
plt.figure(figsize=(10, 8))
imp_coefs.plot(kind="barh", color='green')
plt.title("Top Features Driving House Prices")
plt.xlabel("Coefficient Strength (Impact on Log Price)")
plt.show() 