from sklearn import datasets
import matplotlib.pyplot as plt 

iris = datasets.load_iris()

features = iris.data[:, [0,1,2,3]]
targets = iris.target
targets.reshape(targets.shape[0], -1)
targets.shape
featuresAll = []
iris.feature_names 

for observation in features:
    featuresAll.append([[observation[0]] + observation[1] + observation[2] + observation[3]])

#Plotando o gráfico de dispersão (GD)

plt.scatter(featuresAll, targets, color='red', alpha=1.0)
plt.title('Iris Dataset scatter plot')
plt.xlabel('features')
plt.ylabel('Targets')
plt.show()

#plotando o GD com Dataset Iris (comprimento e largura da sépala)

#achando a relação entre o comprimento e a largura da sépala 
sepal_len = []
sepal_width = []
for feature in features:
    sepal_len.append(feature[0]) #comprimento da sépala
    sepal_width.append(feature[1]) #largura da sépala 

groups = ('Íris-setosa', 'Íris-versicolor', 'Íris-virgínica')
colors = ('Red', 'Blue', 'Pink')
data = ((sepal_len[:50], sepal_width[:50]), (sepal_len[50:100], sepal_width[50:100]), (sepal_len[100:150], sepal_width[100:150]))

for item, color, group in zip(data, colors, groups):
    #item = (sepal_len[:50], sepal_width[:50]), (sepal_len[50:100], sepal_width[50:100]), (sepal_len[100:150], sepal_width[100:150])
    x0, y0 = item 
    plt.scatter(x0, y0, color = color, alpha = 1)
    plt.title('Iris Dataset scatter Plot (Sepal)')

plt.xlabel('Sepal length')
plt.ylabel('Sepal Width')
plt.show()

#plotando o GD com Dataset Iris (comprimento e largura da pétala)

#achando a relação entre o comprimento e a largura da pétala 
petal_len = []
petal_width = []
for feature in features:
    petal_len.append(feature[2]) #comprimento da pétala 
    petal_width.append(feature[3]) #largura da pétala  

groups = ('Íris-setosa', 'Íris-versicolor', 'Íris-virgínica')
colors = ('Red', 'Blue', 'Pink')
data = ((petal_len[:50], petal_width[:50]), (petal_len[50:100], petal_width[50:100]), (petal_len[100:150], petal_width[100:150]))

for item, color, group in zip(data, colors, groups):
    #item = (petal_len[:50], petal_width[:50]), (petal_len[50:100], petal_width[50:100]), (petal_len[100:150], petal_width[100:150])
    x0, y0 = item 
    plt.scatter(x0, y0, color = color, alpha = 1)
    plt.title('Iris Dataset scatter Plot (Petal)')

plt.xlabel('Petal length')
plt.ylabel('Petal Width')
plt.show()