import os
import pandas as pd
import numpy as np
from sklearn import preprocessing, tree, model_selection
import pydot
import io


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

x_train.todense()
#==============================================================================
# matrix([[ 0.,  1.,  0., ...,  0.,  0.,  1.],
#         [ 1.,  0.,  1., ...,  1.,  0.,  0.],
#         [ 1.,  0.,  0., ...,  0.,  0.,  1.],
#         ..., 
#         [ 1.,  0.,  0., ...,  0.,  0.,  1.],
#         [ 0.,  1.,  1., ...,  1.,  0.,  0.],
#         [ 0.,  1.,  0., ...,  0.,  0.,  1.]])
#==============================================================================

np.set_printoptions(threshold=np.nan)

x_train.todense()
# Now it will display whole matrix for more details see in the below example 

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



dot_data = io.StringIO() 
tree.export_graphviz(dt, out_file = dot_data, feature_names = x_train)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
os.chdir("E:\\DS\\Decision tree\\Tree diagrams")
graph.write_pdf("DT_with_OHE1.pdf")




# *************************  Example ******************************************
dct = {"A": ["S","T","U","S","S","S","U","T","T"],
       "B": ["X","Y","X","Y","Y","X","Z","X","Z"],
       "C": ["I","I","J","J","I","J","I","J","J"]
       }

df = pd.DataFrame.from_dict(dct)

#==============================================================================
#    A  B  C
# 0  S  X  I
# 1  T  Y  I
# 2  U  X  J
# 3  S  Y  J
# 4  S  Y  I
# 5  S  X  J
# 6  U  Z  I
# 7  T  X  J
# 8  T  Z  J
#==============================================================================

df1 = df.copy()

df1.A = le.fit_transform(df1.A)
df1.B = le.fit_transform(df1.B)
df1.C = le.fit_transform(df1.C)

df1
#==============================================================================
#    A  B  C
# 0  0  0  0
# 1  1  1  0
# 2  2  0  1
# 3  0  1  1
# 4  0  1  0
# 5  0  0  1
# 6  2  2  0
# 7  1  0  1
# 8  1  2  1
#==============================================================================

x_df1 = ohe.fit_transform(df1[["A","B","C"]]) 

x_df1.todense()
#==============================================================================
# matrix([[ 1.,  0.,  0., ...,  0.,  1.,  0.],
#         [ 0.,  1.,  0., ...,  0.,  1.,  0.],
#         [ 0.,  0.,  1., ...,  0.,  0.,  1.],
#         ..., 
#         [ 0.,  0.,  1., ...,  1.,  1.,  0.],
#         [ 0.,  1.,  0., ...,  0.,  0.,  1.],
#         [ 0.,  1.,  0., ...,  1.,  0.,  1.]])    
#==============================================================================
    
print(x_df1[0,:])
#==============================================================================
#   (0, 6)        1.0
#   (0, 3)        1.0
#   (0, 0)        1.0
#==============================================================================

print(x_df1[2,:])

#==============================================================================
#   (0, 7)        1.0
#   (0, 3)        1.0
#   (0, 2)        1.0
#==============================================================================
  
  
print(x_df1)
#==============================================================================
#   (0, 6)        1.0
#   (0, 3)        1.0
#   (0, 0)        1.0
#   (1, 6)        1.0
#   (1, 4)        1.0
#   (1, 1)        1.0
#   (2, 7)        1.0
#   (2, 3)        1.0
#   (2, 2)        1.0
#   (3, 7)        1.0
#   (3, 4)        1.0
#   (3, 0)        1.0
#   (4, 6)        1.0
#   (4, 4)        1.0
#   (4, 0)        1.0
#   (5, 7)        1.0
#   (5, 3)        1.0
#   (5, 0)        1.0
#   (6, 6)        1.0
#   (6, 5)        1.0
#   (6, 2)        1.0
#   (7, 7)        1.0
#   (7, 3)        1.0
#   (7, 1)        1.0
#   (8, 7)        1.0
#   (8, 5)        1.0
#   (8, 1)        1.0
#==============================================================================

np.set_printoptions(threshold=np.nan)
x_df1.todense()

# A column is split into 3, B --> 3 and c --> 2, Total = 8 columns.
# in each and every column 0 is converted to 1 and 1,2 are converted to 0.
#==============================================================================
# matrix([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
#         [ 0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.],
#         [ 0.,  0.,  1.,  1.,  0.,  0.,  0.,  1.],
#         [ 1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.],
#         [ 1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.],
#         [ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.],
#         [ 0.,  0.,  1.,  0.,  0.,  1.,  1.,  0.],
#         [ 0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
#         [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.]])
#==============================================================================

df1
#==============================================================================
#    A  B  C
# 0  0  0  0
# 1  1  1  0
# 2  2  0  1
# 3  0  1  1
# 4  0  1  0
# 5  0  0  1
# 6  2  2  0
# 7  1  0  1
# 8  1  2  1
#==============================================================================

# ************************* End of Example ************************************
# for more details goto http://stackoverflow.com/questions/15115765/how-to-access-sparse-matrix-elements
 