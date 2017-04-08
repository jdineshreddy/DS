import os
import pandas as pd
#import numpy as np
from sklearn import tree
import pydot
import io

os.getcwd()

os.chdir("E:\\Algorithmica\\Kaggle\\Titanic")

train = pd.read_csv("train.csv")
train.shape
train.info()

# fill missing data of Embarked columns
train.ix[61,'Embarked'] = 'S'
train.loc[61,'Embarked'] = 'S'

train1 = pd.get_dummies(train, columns = ['Pclass', 'Sex', 'Embarked'])
train1.info()

dt = tree.DecisionTreeClassifier()
x_values = train1[['Embarked_C', 'Embarked_S' , 'Embarked_Q','Pclass_1', 'Pclass_2', 'Pclass_3',
                   'Sex_female', 'Sex_male','Fare']]
y_value = train1['Survived']


dt.fit(x_values,y_value)

dot_data = io.StringIO() 
tree.export_graphviz(dt, out_file = dot_data, feature_names = x_values.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
os.chdir("E:\\DS\\Decision tree\\Tree diagrams")
graph.write_pdf("three_dt1.pdf")

  