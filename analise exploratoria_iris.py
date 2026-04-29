# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:46:12 2021

@author: Karla
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import NeighborhoodComponentsAnalysis as nca
from sklearn.pipeline import Pipeline as pipe

#Data import
#df_base = pd.read_csv('./oakland-street-trees.csv')

df_base = pd.read_csv('iris_v2.csv',sep=';')
columns_iris = ['sepal_length','sepal_width','petal_length','petal_width']


print(df_base.info())

print('Estatística básica dos atributos:')
print(df_base.describe())
print('--------------------------------------------------------------')
print('Número de  atrbutos', df_base.shape[1])
print('--------------------------------------------------------------')


#Scatter Plot dos dados
print('--------------------------------------------------------------')

sns.pairplot(df_base)
df_base.plot(kind='scatter', x='species', y='petal_length')
df_base.plot(kind='scatter', x='species', y='petal_width')
# plt.show()

#Histogramas
plt.figure()
df_base['petal_length'].plot(kind='hist', title = 'petal_length1', bins=10, figsize=(12,6), facecolor='blue',edgecolor='black')
# plt.show()

plt.figure()
df_base['petal_width'].plot(kind='hist', title = 'petal_width',  bins=20, figsize=(12,6), facecolor='green',edgecolor='black')
# plt.show()


#Correlações simples
plt.figure()
corr = df_base.select_dtypes(include='number').corr() # plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


#Bar chart for top 5 species of flowers
df_topSpecies = df_base['species'].groupby(df_base['species']).count().sort_values(ascending=False).head(5)
fig = plt.figure(dpi=90)
df_topSpecies.plot.bar()
plt.style.use('seaborn-v0_8-pastel')
plt.title('Most planted trees, by species')
plt.xlabel('Species')
plt.ylabel('Number of flowers')
print("These top 5 species make up for", df_topSpecies.sum(), "of", len(df_base), "trees planted (or", df_topSpecies.sum()/len(df_base),"% of trees).")

plt.show()