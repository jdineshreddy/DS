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

# titanic problem kaggel
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
print(titanic_train1[df_f3])     # passenger 34, 375 having same fare that way no spliting happend

# second analysis for fare between 90.5396, 99.9625 from dt1.pdf file
df_f4 = titanic_train1['Fare'].between(90.5396, 99.9625) 
sum(df_f4)  # print 4(count)
print(titanic_train1.loc[df_f4])  
print(titanic_train1[df_f4]) 