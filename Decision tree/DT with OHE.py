import os
import pandas as pd
#import numpy as np
from sklearn import preprocessing, tree, model_selection


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

# **************************** End of On Hot Encoding *************************

# In Python we use get_dummy for easy process.

dt = tree.DecisionTreeClassifier(random_state = 10)
y_train = train_le['Survived']
dt.fit(x_train,y_train)
dt.score(x_train,y_train) # 0.81144781144781142
model_eve = model_selection.cross_val_score(dt, x_train, y_train, cv = 10)
#==============================================================================
# array([ 0.82222222,  0.77777778,  0.79775281,  0.85393258,  0.86516854,
#         0.79775281,  0.80898876,  0.76404494,  0.82022472,  0.80681818])
#==============================================================================
model_eve.mean() # 0.81146833503575078

# Using Get_dummy we got 0.8171742707978662  accuracy.
