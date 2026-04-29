# -*- coding: utf-8 -*-
"""
Created on Sun May  7 14:48:00 2023

@author: ADM
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import KBinsDiscretizer 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import EqualWidthDiscretiser

# Load dataset
#data = pd.read_csv('houseprice.csv')
data_iris = pd.read_csv('iris.csv', sep=';')

# dividindo o atributo 'sepal_length' em 10 faixas
bins = pd.qcut(data_iris['sepal_length'], 10)
grupos = data_iris['sepal_length'].groupby(bins,observed=False) #agrupa valores de acordo com bins feitas de igual frequencia
# obtendo a media de cada faixa
medias = grupos.mean()
print('-----------------------------------------------')
print("Medias :")
print(medias)
novo_mean_sepal_length = bins.apply(lambda x : medias[x]) #troca cada valor pela media do seu bin(x eh um bin, que serve como indice, pois medias eh um Series)
print('-----------------------------------------------')
print('Velho sepal_length')
print(data_iris['sepal_length'])
data_iris['sepal_length'] = novo_mean_sepal_length
print('-----------------------------------------------')
print('Novo sepal_length')
print(data_iris['sepal_length'])