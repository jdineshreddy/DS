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
train_le.apply(lambda x: sum(x.isnull()))

train_le.Embarked = le.fit_transform(train_le.Embarked)

# **************************** On Hot Encoding ********************************
ohe = preprocessing.OneHotEncoder()
# In Python we use get_dummy fot one hot encoding
# **************************** End of On Hot Encoding *************************
