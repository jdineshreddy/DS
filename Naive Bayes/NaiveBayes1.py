import os
import pandas as pd
from sklearn import preprocessing

os.getcwd()
#change directory
os.chdir('E:\\Algorithmica\\Kaggle\\Titanic')

# read train.csv file
train = pd.read_csv('train.csv')
print(type(train)) #<class 'pandas.core.frame.DataFrame'>

# create another dataframe with pclass, sex, Embarked, Survived columns with only 10 rows.
# modified train data
mod_train = train['Sex']
type(mod_train) # data type of output is pandas.core.series.Series        


# data type of output is pandas.core.series.Series and output is like maleQ,maleC,femaleS.....   
mod_train =    train['Sex'] + train['Embarked']

# output is again a dataframe. Data type of output is pandas.core.frame.DataFrame 
# It is having 3 columns Sex, Embarked, Pclass, Survived with first 10 rows.
mod_train = train[['Sex', 'Embarked', 'Pclass', 'Survived']].head(10) 

# Using label encoder convert strings into numirics
# Ex. here Sex is convert to 0, 1(numerical values) and pclass,embarked to 1,2,3   
le = preprocessing.LabelEncoder() # le is Object of a "transformer" LabelEncoder
mod_train.Sex = le.fit_transform(mod_train.Sex)
mod_train.Embarked = le.fit_transform(mod_train.Embarked)
mod_train.Pclass = le.fit_transform(mod_train.Pclass)

# mod_train dataframe
#==============================================================================
#    Sex  Embarked  Pclass  Survived
# 0    1         2       2         0
# 1    0         0       0         1
# 2    0         2       2         1
# 3    0         2       0         1
# 4    1         2       2         0
# 5    1         1       2         0
# 6    1         2       0         0
# 7    1         2       2         0
# 8    0         2       2         1
# 9    0         0       1         1
#==============================================================================

# Calculating Naive Bayes probubality 
# assuming first row
#P(Survived = 0 /(Sex=1,Embarked=2, Pclass=2))
# from Bayes theorem we can write as 
# posterior probability =  (Likelihood * prior belief/probability)/ Evidence
# Likelihood = P((Sex=1,Embarked=2, Pclass=2)/Survived = 0)
# Prior belief = P(Survived = 0)
# Evidence = P(Sex=1,Embarked=2, Pclass=2), Since evidence is same in all the probability. we are not calculating it.

# According to Naive Bayes all the events are independent,Hence
# Likelihood = P(Sex=1/Survived = 0) * P(Embarked=2/Survived = 0) * P(Pclass=2/Survived = 0)
# posterior probability =  (Likelihood * prior belief/probability)
sex = pd.crosstab(mod_train['Survived'],mod_train['Sex'])
Emb = pd.crosstab(mod_train[ 'Survived'], mod_train['Embarked'])
Pcls = pd.crosstab(mod_train[ 'Survived'], mod_train['Pclass'])  
              

