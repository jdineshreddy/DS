# http://hamelg.blogspot.in/2015/11/python-for-data-analysis-part-19_17.html
import os
import pandas as pd


os.getcwd()
os.chdir('E:\\Algorithmica\\Kaggle\\Titanic')

train = pd.read_csv('train.csv')
train.iloc[1:7]             # Slice rows 1-6

sur_count = pd.crosstab(index = train['Survived'], columns = 'count')
#==============================================================================
# col_0     count
# Survived       
# 0           549
# 1           342
#==============================================================================
sur_count.sum() # count    891
sur_count.shape # (2,1)


pcl_count = pd.crosstab(index = train['Pclass'], columns = 'count')
#==============================================================================
# col_0   count
# Pclass       
# 1         216
# 2         184
# 3         491
#==============================================================================
pcl_count.sum() #count    891
pcl_count.shape # (3, 1)

sex_count = pd.crosstab(index = train['Sex'], columns = 'count')
#==============================================================================
# col_0   count
# Sex          
# female    314
# male      577
#==============================================================================
sex_count.shape # (2,1)

sur_sex = pd.crosstab(index = train['Survived'], columns = train['Sex'])
#==============================================================================
# Sex       female  male
# Survived              
# 0             81   468
# 1            233   109
#==============================================================================

sur_sex.index = ['Dead', 'Survived']
#==============================================================================
# Sex       female  male
# Dead          81   468
# Survived     233   109
#==============================================================================
sur_sex1 = pd.crosstab(index = train['Survived'], columns = train['Sex'], margins = True)
sur_sex1.index = ['Dead', 'Survived', 'ColCount']
sur_sex1.columns = ['female','male','RowCount']
#==============================================================================
#           female  male  RowCount
# Dead          81   468       549
# Survived     233   109       342
# ColCount     314   577       891
#==============================================================================
sur_sex1.ix['ColCount'] # similar to     sur_sex1.iloc[2] 
#==============================================================================
# female      314
# male        577
# RowCount    891
# Name: ColCount, dtype: int64
#==============================================================================
sur_sex1/sur_sex1.ix['ColCount','RowCount'] # dividing each value with 891
#==============================================================================
#             female      male  RowCount
# Dead      0.090909  0.525253  0.616162
# Survived  0.261504  0.122334  0.383838
# ColCount  0.352413  0.647587  1.000000
#==============================================================================
sur_sex1/sur_sex1.ix['ColCount']  # diving with column sum
#==============================================================================
#             female      male  RowCount
# Dead      0.257962  0.811092  0.616162
# Survived  0.742038  0.188908  0.383838
# ColCount  1.000000  1.000000  1.000000
#==============================================================================
#*****************************************************************************************
# diving with row sum
# In Python Usually axis=0 is said to be "column-wise" (and axis=1 "row-wise")
sur_sex1.div(sur_sex1['RowCount'],axis = 0) 
#==============================================================================
#             female      male  RowCount
# Dead      0.147541  0.852459       1.0
# Survived  0.681287  0.318713       1.0
# ColCount  0.352413  0.647587       1.0
#==============================================================================


pcls_sur = pd.crosstab(index = train['Survived'], columns = train['Pclass'])
#==============================================================================
# Pclass      1   2    3
# Survived              
# 0          80  97  372
# 1         136  87  119
#==============================================================================
pcls_sur.sum()  # column vice addition
#==============================================================================
# Pclass
# 1    216
# 2    184
# 3    491
# dtype: int64
#==============================================================================
pcls_sur1 = pd.crosstab(index = train['Survived'], columns = train['Pclass'], margins = True)
#==============================================================================
# Pclass      1    2    3  All
# Survived                    
# 0          80   97  372  549
# 1         136   87  119  342
# All       216  184  491  891
#==============================================================================

pcls_sur1.index = ['Dead', 'Survuved', 'Col_count']
pcls_sur1.columns = ['Class1','Class2','Class3','Row_count']

pcls_sur1
#==============================================================================
#            Class1  Class2  Class3  Row_count
# Dead           80      97     372        549
# Survuved      136      87     119        342
# Col_count     216     184     491        891
#==============================================================================
