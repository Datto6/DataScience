import pandas as pd
dados=pd.read_csv("class_german_credit.csv")
print(dados.head())
print(dados.info())
print(dados.describe())