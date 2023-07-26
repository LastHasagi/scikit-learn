import pandas as pd 
import numpy as np
import time 
from sklearn import datasets 
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix  

iris = datasets.load_iris()
irs = pd.DataFrame(iris.data, columns = iris.feature_names)
irs['class'] = iris.target

#Para começar a usar o knn precisamos fazer pré processamentos 

x = irs.iloc[:, :-1].values
y = irs.iloc[:, 4].values

#Divisão em conjunto de teste e de treino
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
scaler = StandardScaler()  
scaler.fit(x_train)

##Feature Scaling (Valores das features na mesma escala)

#Normalização
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test) 

#Treinamento e Previsões
classifier = KNeighborsClassifier(n_neighbors = 10)  
classifier.fit(x_train, y_train)  

#Avaliando o Algoritmo
y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

#Calcular taxa de erro de 1~40
ansr = input('Deseja calcular a taxa de erro ? [s] [n] ')
if ansr == 's' or 'S':
    error = []

# Calculando erro para valores de K entre 1 e 40
    for i in range(1, 40):  
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error.append(np.mean(pred_i != y_test))
    
    plt.figure(figsize=(12, 6))  
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
        markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')  
    plt.xlabel('K Value')  
    plt.ylabel('Mean Error')

#Para mostrar o gráfico
    plt.show()

else:
    print('Obrigado por usar o código.')
    print('Feito por Alucard')
    time(3)
    exit()