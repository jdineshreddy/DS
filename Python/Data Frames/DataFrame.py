#From dict of Series or dicts
import pandas as pd
d  = {'one' : pd.Series([1,2,3,4,5,6],['a','b','c','d','e','f']),
      'two' : pd.Series([6,7,8,9,0,1],['a','b','c','d','e','g'])}

df = pd.DataFrame(d)

print df
print df.index
print df.columns


# From dict of ndarrays / lists

d1 = {'one' : [1,2,3,4,5,6],
      'two' : [6,7,8,9,0,1]}
df1 = pd.DataFrame(d1)

print df1
print df1.index
print df1.columns

#From a list of dicts

d3 = [{'a' : 1,'b' : 2},{'a' : 5, 'b' : 6, 'c' :7}]
df3 = pd.DataFrame(d3)

print df3
print df3.index
print df3.columns