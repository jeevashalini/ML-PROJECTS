# %%

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt



# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
df =pd.read_csv('dataset.csv') #loading the dataset available

# %%
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.isnull().sum().sort_values(ascending=False)

# %%
missing_val_cols = ["fixed acidity", "pH", "volatile acidity", "sulphates", "citric acid", "residual sugar", "chlorides"]

# %%
for col in missing_val_cols:
    mean = df[col].mean()
    df[col].fillna(mean, inplace=True)
    

# %%
df.isnull().sum()

# %%
df['type'].value_counts()

# %%
df['quality'].value_counts()

# %%
sns.pairplot(df, hue='quality', corner=True)

# %%
sns.catplot(x="type", y="citric acid", kind="box", hue="quality", data=df)

# %%
sns.countplot(x="quality", hue="type", data=df)

# %%
sns.countplot(x="type", hue="quality", data=df)

# %%
corr = df.corr()

# %%
plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True,cmap=sns.diverging_palette(200, 10, as_cmap=True), vmin=-1, vmax=1, linewidths=.5, fmt=".2f")

# %%
df = pd.get_dummies(df, drop_first=True)

# %%
df.head()

# %%
df = df.rename(columns={"type_white": "wine_type"})

# %%
df["wine_quality"] = [1 if x>6 else 0 for x in df.quality]

# %%
df.head()

# %%
y = df["wine_quality"]

# %%
y.value_counts()

# %%
y

# %%
x = df.drop(["quality", "wine_quality"], axis=1)

# %%
x

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
log = pd.DataFrame(columns=["model", "accuracy"])

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
from lib.utils import *
ac=[]

# %%

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(11,activation='relu',input_dim=12))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(X_train,y_train,epochs=6)
ac.append(accuracy_score(model,y_test,sample_weight=0.8)*100)

# %%
from sklearn import linear_model
model = linear_model.Lasso(alpha=0.1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
ac.append(accuracy_score(y_pred,y_test)*100)


# %%

plt.style.use('ggplot')
x=['ANN','Lasso Regression']
 
ax=sns.barplot(x,ac[:2])
ax.set_title('Accuracy comparison')
ax.set_ylabel('Accuracy')
#ax.yaxis.set_major_locator(ticker.LinearLocator())
print("the accuracy of {} is {} and {} is {}".format(x[0],ac[0],x[1],ac[1]))
ax.set_ylim(50,100)

# %%


# %%


# %%



