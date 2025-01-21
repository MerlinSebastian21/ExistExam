#!/usr/bin/env python
# coding: utf-8

# # Genpact Machine Learning Hackathon
# Your client is a meal delivery company which operates in multiple cities. They have various fulfillment centers in these cities for dispatching meal orders to their customers. The client wants you to help these centers with demand forecasting for upcoming weeks so that these centers will plan the stock of raw materials accordingly.
# 
# The replenishment of majority of raw materials is done on weekly basis and since the raw material is perishable, the procurement planning is of utmost importance. Secondly, staffing of the centers is also one area wherein accurate demand forecasts are really helpful. Given the following information, the task is to predict the demand for the next 10 weeks (Weeks: 146-155) for the center-meal combinations in the test set:  
# 
# Historical data of demand for a product-center combination (Weeks: 1 to 145)
# Product(Meal) features such as category, sub-category, current price and discount
# Information for fulfillment center like center area, city information etc.

# In[405]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


# In[323]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')


# In[388]:


# Importing training and testing datasets

df_train = pd.read_csv("f_train.csv")
df_test = pd.read_csv("f_test.csv")


# In[389]:


df_train.head()


# In[386]:


df_train.head()


# In[174]:


df_test.head()


# In[176]:


df_train.shape


# In[178]:


df_test.shape


# In[180]:


df_train.columns


# In[182]:


df_test.columns


# In[184]:


df_train.info()


# In[186]:


df_test.info()


# In[188]:


df_train.describe()


# In[189]:


df_test.describe()


# # Checking for Missing values

# In[193]:


df_train.isna().sum()


# In[195]:


df_test.isna().sum()


# In[197]:


df_train["emailer_for_promotion"].value_counts()


# In[199]:


df_train["homepage_featured"].value_counts()


# # Duplicates rows

# In[202]:


df_train.duplicated().sum()


# In[204]:


df_test.duplicated().sum()


# # Checking Correlation of Columns

# In[207]:


sns.heatmap(df_train.corr(),annot= True)


# In[208]:


# Checking Histogram for data imputation


# In[211]:


num_columns = df_train.columns.tolist()
print("Numerical columns:",num_columns)


num_columns_test = df_test.columns.tolist()
print("Numerical columns of test data:",num_columns_test)


# In[213]:


# Train Data
for col in num_columns:
    plt.hist(df_train[col])
    plt.xlabel(col)
    plt.ylabel('count')
    plt.title('Histogram of {}'.format(col))
    plt.show()


# In[214]:


# Test Data


# In[153]:


for col in num_columns_test:
    plt.hist(df_test[col])
    plt.xlabel(col)
    plt.ylabel('count')
    plt.title('Histogram of {}'.format(col))
    plt.show()


# # Outlier Handling

# In[155]:


df_train.boxplot()
plt.xticks(rotation=90)
plt.show()


# In[156]:


df_test.boxplot()
plt.xticks(rotation=90)
plt.show()


# In[157]:


def remove_outliers(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    df[column_name] = df[column_name].clip(upper=upper_bound)
    df[column_name] = df[column_name].clip(lower=lower_bound)
    return df[column_name]


# In[158]:


for col in num_columns:
  df_train[col] = remove_outliers(df_train, col)


# In[159]:


for col in num_columns_test:
  df_test[col] = remove_outliers(df_test, col)


# In[160]:


df_train.boxplot()
plt.xticks(rotation=90)
plt.show()


# In[161]:


df_test.boxplot()
plt.xticks(rotation=90)
plt.show()


# In[162]:


ID =  pd.DataFrame(df_test['id'])
ID


# In[163]:


df_train = df_train.drop('id',axis = 1)
df_test = df_test.drop('id',axis = 1)


# # Scaling

# In[225]:





# In[227]:


# min max scaling for features having non-gaussian distribution in training dataset
min_max = MinMaxScaler()

df_train[num_columns] = min_max.fit_transform(df_train[num_columns])
df_train


# In[234]:


# min max scaling for features having non-gaussian distribution in testing dataset
df_test[num_columns_test] = min_max.fit_transform(df_test[num_columns_test])
df_test


# In[236]:


# Model Training


# In[242]:


# Seperating Features and labels
X = df_train.drop(['num_orders'],axis =1)
y = df_train['num_orders']
X_test1 = df_test


# In[244]:


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 46)


# # Linear Regression

# In[254]:


model = LinearRegression()
model.fit(X_train, y_train)  


# In[338]:


# Making Predictions  
y_pred = model.predict(X_test)  

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  

# Calculate evaluation metrics  
mae = mean_absolute_error(y_test, y_pred)  
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  

# Print the evaluation metrics  
print(f'Mean Absolute Error (MAE): {mae:.4f}')  
print(f'Mean Squared Error (MSE): {mse:.4f}')  
print(f'R² Score: {r2:.2f}')


# In[343]:


#  prediction on testing data
y_pred1 = model.predict(X_test1)


# #### Polynomial Regression 
# 

# In[341]:


poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f'Mean Squared Error (MSE): {mse_poly: .4f}') 


# # DecisionTreeRegressor

# In[345]:


dt_regressor = DecisionTreeRegressor(max_depth=4, random_state=42)

dt_regressor.fit(X_train, y_train)

# Predict on the train set
y_pred_dt = dt_regressor.predict(X_test)

# Evaluate the model performance
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f"Decision Tree MSE: {mse_dt}")


# In[347]:


# Predict on the test set
y_pred_dt1 = dt_regressor.predict(X_test1)


# In[303]:


#Random Forest


# In[ ]:


rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
rf_regressor.fit(X_train, y_train)


# In[316]:


y_pred_rf = rf_regressor.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest MSE: {mse_rf}")


# In[349]:


y_pred_rf1 = rf_regressor.predict(X_test1)


# # XGBoost

# In[329]:


xg_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
xg_regressor.fit(X_train, y_train)


# In[333]:


y_pred_xg = xg_regressor.predict(X_test)

mse_xg = mean_squared_error(y_test, y_pred_xg)
print(f"XGBoost MSE: {mse_xg}")



# In[354]:


# Mean Squared Error Linear Regression (MSE): 0.0002
# Mean Squared Error (MSE) Polynomial Regression:  0.0002
# Decision Tree MSE: 0.00018965082205803382
# Random Forest MSE: 0.00018894238113835343
# XGBoost MSE: 9.802237881238778e-05


# In[356]:


# Model with Lowest MSE is XGBoost


# In[407]:


# Hyperparameter Tuning


# In[409]:


import xgboost as xgb  
from sklearn.model_selection import GridSearchCV  
from sklearn.metrics import mean_squared_error  
import pandas as pd  
import numpy as np  

# Sample Data (replace with your actual data)  
# X_train, X_test, y_train, y_test should be defined beforehand  
# For demonstration, we'll create dummy data.  
from sklearn.datasets import make_regression  
X, y = make_regression(n_samples=100, n_features=10, noise=0.1)  
X_train, X_test, y_train, y_test = X[:80], X[80:], y[:80], y[80:]  

# Initialize the XGBoost regressor  
xg_regressor = xgb.XGBRegressor()  

# Define the parameter grid for hyperparameter tuning  
param_grid = {  
    'n_estimators': [100, 200],  
    'max_depth': [3, 6, 10],  
    'learning_rate': [0.01, 0.1, 0.2],  
    'subsample': [0.8, 1.0],  
}  

# Set up GridSearchCV  
grid_search = GridSearchCV(estimator=xg_regressor,   
                           param_grid=param_grid,  
                           scoring='neg_mean_squared_error',  
                           cv=5,  # 5-fold cross-validation  
                           verbose=1,  
                           n_jobs=-1)  

# Fit the model  
grid_search.fit(X_train, y_train)  

# Get the best parameters  
best_params = grid_search.best_params_  
print(f"Best parameters: {best_params}")  

# Train the final model using the best parameters  
best_model = xgb.XGBRegressor(**best_params)  
best_model.fit(X_train, y_train)  

# Make predictions  
y_pred_xg = best_model.predict(X_test)  

# Calculate Mean Squared Error  
mse_xg = mean_squared_error(y_test, y_pred_xg)  
print(f"XGBoost MSE: {mse_xg}")


# # Saving Results

# In[395]:


result_df = pd.DataFrame(y_pred1)
result_df


# In[397]:


result_df = pd.concat([ID,result_df],axis = 1)
result_df.rename(columns={0 : 'num_orders'}, inplace=True)


# In[399]:


result_df


# In[401]:


result_df.to_csv("submission_file.csv",index=False)


# In[403]:


saved_file = pd.read_csv('submission_file.csv')
print(saved_file.head())


# In[ ]:




