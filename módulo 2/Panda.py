import numpy as np
import pandas as pd 

#criar um array com numpy 

data = np.array(['a', 'b', 'c', 'd'])

#usar o array para gerar um objeto Series do Pandas

s1 = pd.Series(data)
print(s1)

#s2

data2 = np.array(['a', 'b', 'c', 'd'])
s2 = pd.Series(data2, index = [10, 11, 12, 13])
print(s2)

#criando uma Serie a partir de um dicionário 

data3 = {'a' : 0., 'b' : 1., 'c' : 2, 'd' : 3}
s3 = pd.Series(data3)
print(s3)

#criando um dataframe com uma lista 

data = [1,2,3,4]
df = pd.DataFrame(data)
print(df)

#DF usando lista de listas 

data = [['maria', 10], ['carlos', 19], ['rosa', 35], ['pedro', 26]]
df = pd.DataFrame(data, columns = ['Nome', 'Idade'])
print(df)