# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

# %%
##loading dataset

dataset = pd.read_csv('/Users/shalini/Documents/Datasets/1000_Companies.csv')
#extracting array
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,4].values
dataset.head()

# %%
dataset.info()

# %%
#data visualization
num_dataset = dataset.select_dtypes(include = ['number'])
num_corr = num_dataset.corr()
sns.heatmap(num_corr)

# %%
#preprocesssing it becoz it contains categorical values using encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

# %%
X = X[:, 1:]  ##avoid dummy data

# %%
#splitting data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# %%
#implement linera regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# %%
#predicting test results
Y_pred  = regressor.predict(X_test)
Y_pred 

# %%
#calculate coefficients
print(regressor.coef_)

# %%
#calculate intercepts(slope)
print(regressor.intercept_)

# %%
# Calculating the R squared value
from sklearn.metrics import r2_score
r2_score(Y_test, Y_pred)

# %%
print(classification_report(Y_test, Y_pred))

# %%



