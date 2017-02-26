import pandas as pd

# From dict 
s = pd.Series({'a': 1, 'b': 2})

print s
print s['a']

# From dict with index
s1 = pd.Series({'a':1,'b':2}, index = ['m','n'])
# or
s2 = pd.Series({'a':1,'b':2},['m','n'])
print s1
print s1['m']
print s2
print s2['m']

# From scalar value
s3 = pd.Series(5,['a','b','c','d','e'])
print s3
print s3[::-1]

#from list
s4 = pd.Series([1,2,3,4,5],['a','b','c','d','e'])

print s4
print s4[-1]
print s4[2:4]
print s4[::-1] # reverse 
print s4[::-2] # reverse with skip 2
print s4[0:]

# Vectorized operations and label alignment with Series

s5 = s + s1
print s5 # all the output values will NaN
