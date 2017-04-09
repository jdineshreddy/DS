import os
import pandas as pd
#import numpy as np
from sklearn import preprocessing # LabelEncoder transformer
from sklearn import tree # Decision Tree Estimator
import pydot # for visualization
import io # for visualization

os.getcwd()
os.chdir("E:\\Algorithmica\\Kaggle\\Titanic")

titanic_train = pd.read_csv("train.csv")

# EDA
#print(titanic_train.shape, titanic_train.dtypes, titanic_train.info(),titanic_train.describe())
#print(titanic_train.describe())


# Embarkedd column is having few empty values lets find out and assign some value. because while converting
# categorical columns(Embarked, Pclass, Sex) data type(object) to data type(int64) there should not be any 
# null values.
pd.crosstab(titanic_train['Embarked'], columns = 'count') # to find the count of embarked values 
sum(titanic_train['Pclass'].isnull()) # gives 0 value because there is no missing data 
sum(titanic_train.Embarked.isnull()) # gives 2 value because there are 2 missing rows
titanic_train.apply(lambda x : sum(x.isnull())) # it gives which row is having missing data
titanic_train.Embarked[titanic_train['Embarked'].isnull()] = 'S' # assigining S because it is having highest count


titanic_train1 = titanic_train.copy()
le = preprocessing.LabelEncoder() # le is Object of a "transformer" LabelEncoder
# converting Embarked, Pclass, Sex data type(object) columns to data type(int64) columns to categorical data types.
# here Sex is convert to 0, 1(numerical values) and pclass,embarked to 1,2,3   
titanic_train1.Sex =  le.fit_transform(titanic_train1.Sex)
titanic_train1.Embarked = le.fit_transform(titanic_train1.Embarked)
titanic_train1.Pclass = le.fit_transform(titanic_train1.Pclass)


# Model Building(Decision Tree)
dt = tree.DecisionTreeClassifier()
# df.fit(x,y) here x = all columns which are used for analysis, y = target column
# so first we haveto define x, y
x_value = titanic_train1[['Fare', 'Pclass']] # here i am assuming only 2 column for practice
y_value = titanic_train1.Survived
dt.fit(x_value,y_value)

#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(dt, out_file = dot_data, feature_names = x_value.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
os.chdir("E:\\DS\\Decision tree\\Tree diagrams")
graph.write_pdf("DT_with_Label_Encoder1.pdf")


# display range of rows in DataFrame
dis_f = titanic_train1.Fare.between(69.425,74.375) #returns boolean
dis_f2 = ( titanic_train1.Fare >= 69.425) & (titanic_train1.Fare <= 74.375) # both are same

print(sum(dis_f), sum(dis_f2), type(dis_f), type(dis_f2)) # displays counts and both returns same value

print(titanic_train1.loc[dis_f2]) # print those rows which satisfies the above condition
print(titanic_train1[dis_f2])  # same as above  

# in dt1.pdf file at the bottom we have only 2 samples between 82.6646, 82.0145 fare. print those columns

df_f3 = titanic_train1.Fare.between(82.0145,82.6646)  
sum(df_f3) # print 2(count)
print(titanic_train1.loc[df_f3])
print(titanic_train1[df_f3])     # passenger 34, 375 having same fare and one9340 is Survived and other(375) is non Survived

# second analysis for fare between 90.5396, 99.9625 from dt1.pdf file
df_f4 = titanic_train1['Fare'].between(90.5396, 99.9625) 
sum(df_f4)  # print 4(count)
print(titanic_train1.loc[df_f4])  
print(titanic_train1[df_f4])   
# passenger 291, 484 having 91.0792 fare and  520,820 having 93.5000 fare.here spliting does not happened.because all are Survived.


     
    

