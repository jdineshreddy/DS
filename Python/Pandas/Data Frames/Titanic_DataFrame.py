import os
import pandas as pd
import numpy as np

os.getcwd()
os.chdir("E:\\Algorithmica\\Kaggle\\Titanic")
titanic_train = pd.read_csv("train.csv")
# titanic problem kaggel
# display range of rows in DataFrame
dis_f = titanic_train.Fare.between(69.425,74.375) #returns boolean
dis_f2 = ( titanic_train.Fare >= 69.425) & (titanic_train.Fare <= 74.375) # both are same

print(sum(dis_f), sum(dis_f2), type(dis_f), type(dis_f2)) # displays counts and both returns same value

dis_f_rows = titanic_train.loc[dis_f2]
dis_f2_rows = titanic_train[dis_f2]
print(dis_f_rows) # print those rows which satisfies the above condition
print(dis_f2_rows)  # same as above  

# Display size
dis_f_rows.size    # 180
dis_f2_rows.size   # 180  
     
# in dt1.pdf file at the bottom we have only 2 samples between 82.6646, 82.0145 fare. print those columns

df_f3 = titanic_train.Fare.between(82.0145,82.6646)  
sum(df_f3) # print 2(count)
print(titanic_train.loc[df_f3])
print(titanic_train[df_f3])     # passenger 34, 375 having same fare that way no spliting happend

# second analysis for fare between 90.5396, 99.9625 from dt1.pdf file
df_f4 = titanic_train['Fare'].between(90.5396, 99.9625) 
sum(df_f4)  # print 4(count)
print(titanic_train.loc[df_f4])  
print(titanic_train[df_f4]) 


Emb_miss = titanic_train[pd.isnull(titanic_train.Embarked)]
type(Emb_miss)  # pandas.core.frame.DataFrame

Emb_miss1 = titanic_train[pd.isnull(titanic_train['Embarked'])]
type(Emb_miss1)  # pandas.core.frame.DataFrame

# Both Emb_miss and Emb_miss1 will give same output
#==============================================================================
#  PassengerId  Survived  Pclass                                       Name  \
# 61            62         1       1                        Icard, Miss. Amelie   
# 829          830         1       1  Stone, Mrs. George Nelson (Martha Evelyn)   
# 
#         Sex   Age  SibSp  Parch  Ticket  Fare Cabin Embarked  
# 61   female  38.0      0      0  113572  80.0   B28      NaN  
# 829  female  62.0      0      0  113572  80.0   B28      NaN     
#==============================================================================
Emb_miss.size
Emb_miss1.size
# Both are giving same size as 24

    
Emb_miss2 = titanic_train[pd.isnull(titanic_train[['Embarked']])]
type(Emb_miss2)  # pandas.core.frame.DataFrame

# It gives whole DataFrame with NaN
#==============================================================================
# PassengerId  Survived  Pclass Name  Sex  Age  SibSp  Parch Ticket  Fare  \
# 0            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 1            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 2            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 3            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 4            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 5            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 6            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 7            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 8            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN 
# .
# .
# .   
#==============================================================================
 

em = titanic_train['Embarked']            # type(em)   =====>   pandas.core.series.Series
em1 = titanic_train[['Embarked']]         # type(em1)  =====>   pandas.core.frame.DataFrame
em2 = titanic_train.Embarked              # type(em2)  =====>   pandas.core.series.Series
    
Emb_miss3 = titanic_train[titanic_train.Embarked.isnull()]
# Emb_miss3 also gives same output
Emb_miss3.size   # 24

Emb_miss4 = titanic_train.iloc[titanic_train.Embarked.isnull()]
#  iLocation based boolean indexing on an integer type is not available

Emb_miss5 = titanic_train.loc[titanic_train.Embarked.isnull()]
Emb_miss5.size
#  Emb_miss5 also gives same output with same size

Emb_miss6 = titanic_train[pd.isnull(titanic_train[['Embarked']]).any(axis=1)]
Emb_miss6.size  
# Emb_miss6 also gives same output with same size

Emb_miss7 = titanic_train[pd.isnull(titanic_train[['Embarked']])]
# It gives whole DataFrame with NaN
#==============================================================================
# PassengerId  Survived  Pclass Name  Sex  Age  SibSp  Parch Ticket  Fare  \
# 0            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 1            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 2            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 3            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 4            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 5            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 6            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 7            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN   
# 8            NaN       NaN     NaN  NaN  NaN  NaN    NaN    NaN    NaN   NaN 
# .
# .
# .   
#==============================================================================


sim_data6 = titanic_train[(titanic_train.Sex == 'female') & (titanic_train.Pclass == 1)
                           & (titanic_train.Survived == 1) & (titanic_train.SibSp == 0) 
                           & (titanic_train.Parch == 0)]
sim_data6.shape # (33, 12) i.e. 3 rows. Again filter

#  Dataframe characters count in column
sim_data6.Embarked.value_counts()
#==============================================================================
# C    17
# S    14
# Name: Embarked, dtype: int64
#==============================================================================

sim_data6.Embarked.value_counts().max()
# 17

