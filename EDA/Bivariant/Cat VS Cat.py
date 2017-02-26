import os
import pandas as pd
import numpy as np
import seaborn as sns

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:\Algorithmica\Kaggle\Titanic")

titanic_train = pd.read_csv("train.csv")

titanic_train['Survived'] = titanic_train['Survived'].astype('category')
titanic_train['Pclass'] = titanic_train['Pclass'].astype('category')
titanic_train['Sex'] = titanic_train['Sex'].astype('category')
titanic_train['Embarked'] = titanic_train['Embarked'].astype('category')
titanic_train.info()
titanic_train.describe()

#explore bivariate relationships: categorical vs categorical 

# univariate 
pd.crosstab(index=titanic_train["Pclass"], columns="count")
sns.countplot(x='Survived',data=titanic_train)

# Bivariate
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Sex'])
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Sex'], margins=True)
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Pclass'], margins=True)


sns.factorplot(x="Sex",hue = "Survived", data=titanic_train, kind="count", size=5)
sns.factorplot(x="Survived", hue="Sex", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", size=6)

#explore bivariate relationships: categorical vs continuous 
sns.factorplot(x="Fare", row="Survived", data=titanic_train, kind="box", size=6)

sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.distplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.boxplot, "Fare").add_legend()