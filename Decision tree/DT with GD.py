import os
import pandas as pd
#import numpy as np
from sklearn import tree, model_selection
import pydot
import io


os.getcwd()

os.chdir("E:\\Algorithmica\\Kaggle\\Titanic")

train = pd.read_csv("train.csv")
train.shape
train.info()

# what get_dummy will return if it has missing data

train_miss = pd.get_dummies(train, columns = ['Pclass', 'Sex', 'Embarked'])
train_miss.info()
train_miss.loc[61]
train_miss.iloc[829]
# Passenger Id ===> 62, 830 are having missing data(NaN value in Embarked column)
# but when we convvert this to dupllicates using "get_dummies" it is returning 0.

#==============================================================================
# PassengerId                                          830
# Survived                                               1
# Name           Stone, Mrs. George Nelson (Martha Evelyn)
# Age                                                   62
# SibSp                                                  0
# Parch                                                  0
# Ticket                                            113572
# Fare                                                  80
# Cabin                                                B28
# Pclass_1                                               1
# Pclass_2                                               0
# Pclass_3                                               0
# Sex_female                                             1
# Sex_male                                               0
# Embarked_C                                             0
# Embarked_Q                                             0
# Embarked_S                                             0
# Name: 829, dtype: object 
#==============================================================================


# fill missing data of Embarked columns
train.ix[61,'Embarked'] = 'S'
train.loc[829,'Embarked'] = 'S' 
# Set value is much faster then the 'ix'  and 'loc'
# As of now there is no option to set multiple values at a time.
# We need to iterate to do.
train.set_value(61, 'Embarked', 'S')
train.set_value(829, 'Embarked', 'S')
# instead of writing two times we can write in a single line as
train.Embarked[train.Embarked.isnull()] = 'S'
train.apply(lambda x : sum(x.isnull()))

train1 = pd.get_dummies(train, columns = ['Pclass', 'Sex', 'Embarked'])
train1.info()

dt = tree.DecisionTreeClassifier()
x_values = train1[['Embarked_C', 'Embarked_S' , 'Embarked_Q','Pclass_1', 'Pclass_2', 'Pclass_3',
                   'Sex_female', 'Sex_male','Fare']]
y_value = train1['Survived']


dt.fit(x_values,y_value)




# ****************** Explain Random_state = some int  *************************

dt.score(x_values,y_value) # 0.90909090909090906
model_eve = model_selection.cross_val_score(dt, x_values, y_value, cv = 10)
#==============================================================================
# array([ 0.78888889,  0.75555556,  0.7752809 ,  0.82022472,  0.8988764 ,
#         0.82022472,  0.83146067,  0.80898876,  0.83146067,  0.82954545])
#==============================================================================
model_eve.mean() # 0.81605067529224828

# Run the above code again and observe the mean
model_eve1 = model_selection.cross_val_score(dt, x_values, y_value, cv = 10)
#==============================================================================
# array([ 0.78888889,  0.75555556,  0.7752809 ,  0.82022472,  0.88764045,
#         0.82022472,  0.83146067,  0.83146067,  0.83146067,  0.82954545])
#==============================================================================
model_eve1.mean() # 0.8171742707978662  Which is different from the model_eve.mean()

# Again run the above code and observe the change in mean
model_eve2 = model_selection.cross_val_score(dt, x_values, y_value, cv = 10) 
#==============================================================================
# array([ 0.78888889,  0.75555556,  0.7752809 ,  0.82022472,  0.8988764 ,
#         0.82022472,  0.83146067,  0.83146067,  0.83146067,  0.82954545])
#==============================================================================
model_eve2.mean() # 0.81829786630348411 
# Which is different from the model_eve.means() and model_eve1.mean()
# So each and every time we are gettin different mean so to overcome that we use random_state = some integer

dt1 = tree.DecisionTreeClassifier(random_state = 10)
dt1.fit(x_values,y_value)
model_eve3 = model_selection.cross_val_score(dt1, x_values, y_value, cv =10)
#==============================================================================
# array([ 0.78888889,  0.75555556,  0.78651685,  0.82022472,  0.8988764 ,
#         0.82022472,  0.83146067,  0.80898876,  0.83146067,  0.82954545])
#==============================================================================
model_eve1.mean() # 0.8171742707978662

# Again run the above code and see the difference in mean
model_eve4 = model_selection.cross_val_score(dt1, x_values, y_value, cv =10)
#==============================================================================
# array([ 0.78888889,  0.75555556,  0.78651685,  0.82022472,  0.8988764 ,
#         0.82022472,  0.83146067,  0.80898876,  0.83146067,  0.82954545])
#==============================================================================
model_eve4.mean() # 0.8171742707978662   
# Each and every time we will get thw same average value.           

# ******************************  END  ****************************************





dot_data = io.StringIO() 
tree.export_graphviz(dt, out_file = dot_data, feature_names = x_values.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
os.chdir("E:\\DS\\Decision tree\\Tree diagrams")
graph.write_pdf("DT_with_GD1.pdf")

  