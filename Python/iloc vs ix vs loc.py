# pandas iloc vs ix vs loc explanation?
# http://stackoverflow.com/questions/31593201/pandas-iloc-vs-ix-vs-loc-explanation

# loc works on labels in the index.
# iloc works on the positions in the index (so it only takes integers).
# ix usually tries to behave like loc but falls back to behaving like iloc if the label is not in the index.
#It's important to note some subtleties that can make ix slightly tricky to use:

#if the index is of integer type, ix will only use label-based indexing and not fall back to position-based indexing. If the label is not in the index, an error is raised.
#if the index does not contain only integers, then given an integer, ix will immediately use position-based indexing rather than label-based indexing. If however ix is given another type (e.g. a string), it can use label-based indexing.
#To illustrate the differences between the three methods, consider the following Series:

import pandas as pd
import numpy as np
s = pd.Series(np.nan, index=[49,48,47,46,45, 1, 2, 3, 4, 5])
s
#==============================================================================
# 49   NaN
# 48   NaN
# 47   NaN
# 46   NaN
# 45   NaN
# 1    NaN
# 2    NaN
# 3    NaN
# 4    NaN
# 5    NaN
#==============================================================================
# Then s.iloc[:3] returns the first 3 rows (since it looks at the position) and s.loc[:3] returns the first 8 rows (since it looks at the labels):

s.iloc[:3]
#==============================================================================
# 49   NaN
# 48   NaN
# 47   NaN
#==============================================================================

s.loc[:3]
#==============================================================================
# 49   NaN
# 48   NaN
# 47   NaN
# 46   NaN
# 45   NaN
# 1    NaN
# 2    NaN
# 3    NaN
#==============================================================================

s.ix[:3] # the integer is in the index so s.ix[:3] works like loc
#==============================================================================
# 49   NaN
# 48   NaN
# 47   NaN
# 46   NaN
# 45   NaN
# 1    NaN
# 2    NaN
# 3    NaN
# Notice s.ix[:3] returns the same Series as s.loc[:3] since it looks for the label first rather than going by position (and the index is of integer type).
# 
#==============================================================================
#What if we try with an integer label that isn't in the index (say 6)?

#Here s.iloc[:6] returns the first 6 rows of the Series as expected. However, s.loc[:6] raises a KeyError since 6 is not in the index.

s.iloc[:6]
#==============================================================================
# 49   NaN
# 48   NaN
# 47   NaN
# 46   NaN
# 45   NaN
# 1    NaN
#==============================================================================
s.loc[:6]
#KeyError: 6

s.ix[:6]
#KeyError: 6
#As per the subtleties noted above, s.ix[:6] now raises a KeyError because it tries to work like loc but can't find a 6 in the index. Because our index is of integer type it doesn't fall back to behaving likeiloc.

#If, however, our index was of mixed type, given an integer ix would behave like iloc immediately instead of raising a KeyError:

s2 = pd.Series(np.nan, index=['a','b','c','d','e', 1, 2, 3, 4, 5])

# *****************************************************
s.index.is_mixed()
# False
s2.index.is_mixed() # index is mix of types
#True
# *****************************************************

s2.ix[:6] # behaves like iloc given integer
#==============================================================================
# a NaN
# b NaN
# c NaN
# d NaN
# e NaN
# 1 NaN
# Keep in mind that ix can still accept non-integers and behave like loc:
#==============================================================================

s2.ix[:'c'] # behaves like loc given non-integer
#==============================================================================
# a NaN
# b NaN
# c NaN
#==============================================================================


# General advice: if you're only indexing using labels, or only indexing using integer positions, stick with loc or iloc to avoid unexpected results.

# If however you have a DataFrame and you want to mix label and positional index types, ix lets you do this:

df = pd.DataFrame(np.arange(25).reshape(5,5), 
                  index=list('abcde'),
                  columns=['x','y','z', 8, 9])
df
#==============================================================================
#     x   y   z   8   9
# a   0   1   2   3   4
# b   5   6   7   8   9
# c  10  11  12  13  14
# d  15  16  17  18  19
# e  20  21  22  23  24
# Using ix, we can slice the rows by label and the columns by position (note that for the columns, ix default to position-based slicing since the label 4 is not a column name):
# 
#==============================================================================
df.ix[:'c', :4]
#==============================================================================
#     x   y   z   8
# a   0   1   2   3
# b   5   6   7   8
# c  10  11  12  13
# 
#==============================================================================
