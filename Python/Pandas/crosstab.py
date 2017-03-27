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

