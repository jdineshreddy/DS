import os
import pandas as pd
os.getcwd()
os.chdir('E:\\Algorithmica\\git\\Dinesh\\DS\\Assignments\\Assignments 1')
rainfall = open("rainfall.dat")
lst = []
for line in rainfall:
    lst += [line.split()]    
rain = pd.DataFrame(lst)
rain.shape
rain.columns
rain.iloc[1][3]
rain.iloc[1]
rain["daily"] = 0
tmp_lst = []
for i in range(0,len(rain)):
    temp = 0
    for j in rain.iloc[i][3:-1]:
        temp += int(j)
    tmp_lst = tmp_lst + [temp]
rain["daily"] = tmp_lst             
print rain    
    