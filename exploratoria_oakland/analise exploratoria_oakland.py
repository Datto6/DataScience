# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:46:12 2021

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
df_base = pd.read_csv('oakland-street-trees.csv')

print(df_base.info())

print('Estatística básica dos atributos:')
print(df_base.describe())


#Bar chart for top 20 species of trees
#Subsampling data into a pandas series of top planted tree species
df_topSpecies = df_base['SPECIES'].groupby(df_base['SPECIES']).count().sort_values(ascending=False).head(20)
fig = plt.figure(dpi=90)
df_topSpecies.plot.bar()
plt.style.use('seaborn-v0_8-pastel')
plt.title('Most planted trees, by species')
plt.xlabel('Species')
plt.ylabel('Number of trees')

print("These top 20 species make up for", df_topSpecies.sum(), "of", len(df_base), "trees planted (or", df_topSpecies.sum()/len(df_base),"% of trees).")
plt.show()