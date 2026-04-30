# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 05:54:33 2023

@author: ADM
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #Funcao para dividir a base de dados


#%matplotlib inline

# - Original data is seperated by delimiter " ; " in given dataset
# - " .head() " returns first five observations of the dataset#
#wine
# - dataset comprises of 4898 observations and 12 chracteriestics 
# - out of which one is dependent variable and rest 11 are independent variables - physicochemical characteristics
# df = pd.read_csv('winequality-white.csv',sep=';')
# columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density' ,'pH',
#  'sulphates', 'alcohol', 'quality']

#iris
#Target variable/Dependent variable is discrete and categorical in nature
# - "species" are virginica, versicolor and setosa
#iris
iris = pd.read_csv('iris.csv',sep=';')
columns_iris = ['sepal_length','sepal_width','petal_length','petal_width']


print('-----------------------------------------------')
print("Head da base de dados:")
print(iris.head())


print('-----------------------------------------------')
print("Tamanho da base de dados:")
print(iris.shape )

# - Label of each column
print('-----------------------------------------------')
print("Rótulos dos atributos:")
print(iris.columns.values)

print('-----------------------------------------------')
print("Quantitativo de Rótulos:")
print(iris['species'].value_counts())

print('-----------------------------------------------')
print("Informações: ")
print(iris.info() )
# - Data has only float and integer values
# - No variable column has null/missing values
print('-----------------------------------------------')
print("Descrição: ")
descricao=iris.describe() 
for i in columns_iris:
    #print(descricao.i)
    print(descricao[i])
    
    
#Key Observations - 
#wine
# - Mean value is less than median value of each column represented 
# by 50%(50th percentile) in index column.
# - Notably large differnece in 75th %tile and max values of predictors
# "residual sugar","free sulfur dioxide","total sulfur dioxide"
# - Thus observations 1 and 2 suggests that there are extreme values-Outliers in our dataset
print('-----------------------------------------------')
print("quantidade de valores únicos nos atributos: ")
#iris
print(iris.species.unique() )
#wine
#print(df.quality.unique() )

# - Target variable/Dependent variable is discrete and categorical in nature.
# - "quality" score scale ranges from 1 to 10;where 1 being poor and 10 being the best.
# - 1,2 & 10 Quality ratings are not given by any obseravtion.Only scores 
# obtained are between 3 to 9.

#opcao para remover dados duplicados
iris_sem_duplicados=iris.drop_duplicates()
print("Quantitativo de Rótulos - sem duplicados:")
print(iris_sem_duplicados['species'].value_counts())


#PYTHON — Amostra Simples
print('-----------------------------------------------')
iris_amostra_simples=iris.sample(100, replace=False)
print("Quantitativo de Rótulos Amostra Simples:")
print(iris_amostra_simples['species'].value_counts())

#PYTHON — Amostra Estratificada
#Amostram com 50 por cento de cada Classe
#iloc -> Buscar parte específica da base de dados

X, _, y, _ = train_test_split(iris.iloc[:, 0:4], iris.iloc[:, 4],
test_size = 0.5, stratify = iris.iloc[:,4])
print('-----------------------------------------------')
print("Quantitativo de Rótulos Amostra Estratificada:")
print(y.value_counts())

