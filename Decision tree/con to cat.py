import os
import pandas as pd
import numpy as np


os.getcwd()

os.chdir("E:\\Algorithmica\\Kaggle\\Titanic")

train = pd.read_csv("train.csv")
train.shape
train.info()

# fill missing data of Embarked column

em = train['Embarked']            #type(em)   =====>   pandas.core.series.Series
em1 = train[['Embarked']]         # type(em1) =====>   pandas.core.frame.DataFrame

em_miss = train[pd.isnull(train[['Embarked']]).any(axis=1)]
#  OR Emb_miss3 = titanic_train[titanic_train.Embarked.isnull()]
# OR  Emb_miss5 = titanic_train.loc[titanic_train.Embarked.isnull()]
#==============================================================================
#      PassengerId  Survived  Pclass                          Name  \
# 61            62         1       1                        Icard, Miss. Amelie   
# 829          830         1       1  Stone, Mrs. George Nelson (Martha Evelyn)   
# 
#         Sex   Age  SibSp  Parch  Ticket  Fare Cabin Embarked  
# 61   female  38.0      0      0  113572  80.0   B28      NaN  
# 829  female  62.0      0      0  113572  80.0   B28      NaN  
# 
#==============================================================================
type(em_miss) # pandas.core.frame.DataFrame
em_miss.size

#   train.notnull() 
#   train.isnull()

# Try to find similar columns which satisfies 61 and 829 columns
sim_data = train[train.Fare == 80] # only 61 and 829 columns have this data
sim_data2 = train[train.Ticket == '113572'] # only 61 and 829 columns have this data
sim_data3 = train[(train.Sex == 'female') & (train.Pclass == 1)] #it has more data(94 rows) so filter it
sim_data4 = train[(train.Sex == 'female') & (train.Pclass == 1) & (train.Survived == 1)] 
#it has more data(91 rows) so filter it
sim_data5 = train[(train.Sex == 'female') & (train.Pclass == 1) & (train.Survived == 1)
                 & (train.SibSp == 0)] 
sim_data5.shape # (48, 12) i.e. 48 rows. Again filter
sim_data6 = train[(train.Sex == 'female') & (train.Pclass == 1) & (train.Survived == 1)
                 & (train.SibSp == 0) & (train.Parch == 0)]
sim_data6.shape # (33, 12) i.e. 3 rows. Again filter

#  Dataframe characters count in column
sim_data6.Embarked.value_counts()
#==============================================================================
# C    17
# S    14
# Name: Embarked, dtype: int64
#==============================================================================

train.Embarked.value_counts()
#==============================================================================
# S    644
# C    168
# Q     77
# Name: Embarked, dtype: int64
#==============================================================================


sim_data6.Embarked.value_counts().max()
# 17

sim_data7 = train[(train.Sex == 'female') & (train.Pclass == 1) & (train.Survived == 1)
                 & (train.SibSp == 0) & (train.Parch == 0) & (train.Cabin == 'B28')]
sim_data7.shape  #  (2, 12)

train[train.Embarked.isnull()].Embarked = 'S'
#==============================================================================
# C:\Users\dines\Anaconda3\lib\site-packages\pandas\core\generic.py:2773: SettingWithCopyWarning: 
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
# 
# See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
#   self[name] = value
#==============================================================================

train.iloc[train.Embarked.isnull()].Embarked = 'S'
# iLocation based boolean indexing on an integer type is not available
train.ix[train.Embarked.isnull()].Embarked = 'S'
# A value is trying to be set on a copy of a slice from a DataFrame.
train.loc[train.Embarked.isnull()].Embarked = 'S'  
#==============================================================================
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
#==============================================================================
train.iloc[61,'Embarked'] = 'S'
#==============================================================================
# Location based indexing can only have [integer, integer slice (START point
# is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types
#==============================================================================
# both works for assigning value
train.ix[61,'Embarked'] = 'S'
train.loc[61,'Embarked'] = 'S'
train.iloc[61]
train.shape  # (891, 12)
train1 = pd.get_dummies(train, columns = ['Pclass', 'Sex', 'Embarked'])
train1.shape #(891, 17)
train1.apply(lambda x: sum(x.isnull()))
train1.info()
train1.describe()     