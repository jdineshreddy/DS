import os
import pandas as pd
import seaborn as sns

# returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:\Algorithmica\Kaggle\Titanic")

titanic_train = pd.read_csv("train.csv")

titanic_train.describe()
titanic_train.info()

titanic_train['Survived'] = titanic_train['Survived'].astype('category')
titanic_train['Pclass'] = titanic_train['Pclass'].astype('category')
titanic_train['Sex'] = titanic_train['Sex'].astype('category')
titanic_train['Embarked'] = titanic_train['Embarked'].astype('category')


#explore univariate categorical feature
titanic_train['Survived'].describe()
v1= titanic_train['Survived'].value_counts()
v2 = pd.crosstab(index=titanic_train["Survived"], columns="count")
print v1
print v2
type(v1)
# data type od crosstab is DataFrame
type(v2)
pd.crosstab(index=titanic_train["Pclass"], columns="count")  
pd.crosstab(index=titanic_train["Sex"],  columns="count")
titanic_train['Sex'].value_counts()

#explore univariate categorical features visually
sns.countplot(x='Survived',data=titanic_train)
sns.countplot(x='Pclass',data=titanic_train)
