import pandas as pd
import matplotlib.pyplot as plt
dados=pd.read_csv("class_german_credit.csv")
# print(dados.head())
# print(dados.info())
print(dados.describe())
for column in dados.select_dtypes(include=['number']):
    plt.figure()
    dados[column].plot(kind='hist', bins=20, title='Data Distribution', xlabel=column)


df_sex = dados['Sex'].groupby(dados['Sex']).count().sort_values(ascending=False) #agrupa a coluna sexo pelos valores de sexo, conta freq, ordem descendente
fig = plt.figure(dpi=90)
df_sex.plot.bar()
plt.style.use('seaborn-v0_8-pastel')
plt.title('Distr Sexo')
plt.xlabel('Sexo')
plt.ylabel('Numero de pessoas')
print(df_sex.head())

df_risk = dados['Risk'].groupby(dados['Risk']).count().sort_values(ascending=False) #distribuicao de resultados desigual--> o que fazer?
fig = plt.figure(dpi=90)
df_risk.plot.bar()
plt.style.use('seaborn-v0_8-pastel')
plt.title('Distr Result')
plt.xlabel('Result')
plt.ylabel('Numero de pessoas')
print(df_risk.head())
plt.show()