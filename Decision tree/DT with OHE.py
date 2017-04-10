import os
import pandas as pd
#import numpy as np
from sklearn import preprocessing


os.getcwd()
os.chdir("E:\\Algorithmica\\Kaggle\\Titanic")

train = pd.read_csv("train.csv")


#*********************  Label Encoder Before filling missing data *************
# Now try to convert using Label Encode and One Hot Encoding
le = preprocessing.LabelEncoder()
train_le = train.copy()
train_le.info()
train_le.Sex = le.fit_transform(train_le.Sex)
train_le.Pclass = le.fit_transform(train_le.Pclass)
train_le.Embarked = le.fit_transform(train_le.Embarked)
# Returns Error message "not supported between instances of 'str' and 'float' "
# because of having missing data. 
# *************  End of Label Encoder Before filling missing data *************

# first fill the missing data 
train_le.set_value(61, 'Embarked', 'S')
train_le.set_value(829, 'Embarked', 'S')

# To set multiple rows we need to iterate.
Emb_nan = train_le.Embarked[train_le.Embarked.isnull()]
Emb_nan.index  # Int64Index([61, 829], dtype='int64')
for i in Emb_nan.index:
    train_le.set_value(i,"Embarked", "S")


train_le.apply(lambda x: sum(x.isnull()))
# Now it's not displayed any error message.

train_le.Embarked = le.fit_transform(train_le.Embarked)


# **************************** On Hot Encoding ********************************
ohe = preprocessing.OneHotEncoder()
x_train = ohe.fit_transform(train_le[["Sex","Embarked","Pclass"]])
x_train
#==============================================================================
# <891x8 sparse matrix of type '<class 'numpy.float64'>'
# 	with 2673 stored elements in Compressed Sparse Row format> 
#==============================================================================

# In Python we use get_dummy for easy process.
# **************************** End of On Hot Encoding *************************
