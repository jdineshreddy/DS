import os
import pandas as pd
import numpy as np
import seaborn as sns
#returns current working directory
os.getcwd()

#changes working directory
os.chdir("E:\\Algorithmica\\Kaggle\\Titanic")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()
titanic_train.describe()

#columns 
titanic_train.columns
#index
titanic_train.index
titanic_train.iloc[9][3]
titanic_train.loc[9][3]
titanic_train.values
titanic_train.values.sum()
for i in range(0,5):
   print i 
a = [1,2,3,4,5,6]
print a[2:-1] 

titanic_train["Survived"] = titanic_train["Survived"].astype("category")
titanic_train["Sex"] = titanic_train["Sex"].astype("category")
titanic_train["Pclass"] = titanic_train["Pclass"].astype("category")
titanic_train["Embarked"] = titanic_train["Embarked"].astype("category")


#explore bivariate relationships: continuous vs continuous 
np.cov(titanic_train["SibSp"],titanic_train["Parch"])
np.corrcoef(titanic_train["SibSp"],titanic_train["Parch"])
sns.jointplot(x="SibSp", y="Parch", data = titanic_train)
sns.jointplot(x="SibSp", y="Parch", data=titanic_train, kind='kde')
sns.pairplot(titanic_train)


v1 = [10,12,15,20,22,25]
v2 = [20,22,25,27,32,35]
v3 = [35,32,30,29,27,26]
np.cov(v1,v2)
np.corrcoef(v1,v2)
np.cov(v1,v3)
np.corrcoef(v1,v3)
df = pd.DataFrame({'v1':v1,'v2':v2,
'v3':v3})
sns.jointplot(x="v1", y="v3", data=df)
sns.jointplot(x='v1', y='v2', data=df)