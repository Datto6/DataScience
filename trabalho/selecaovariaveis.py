import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy import stats
import seaborn as sns
preprocessor = ColumnTransformer([
    # Sex -> binary
    ('sex',OrdinalEncoder(categories=[['female', 'male']]), ['Sex']),

    # Housing -> ordinal
    ( 'housing',OrdinalEncoder(categories=[['free','rent', 'own']]),['Housing']),

    # Saving accounts
    ('saving',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,categories=[['little','moderate','rich','quite rich']]),['Saving accounts']),
    # Checking account
    ('checking',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,categories=[['little','moderate','rich']]),['Checking account']),
    #Coisas de discretizacao, nao funcionou
    # ('age_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'), ['Age']),

    # ('duration_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'),['Duration']),

    # ('credit_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'),['Credit amount'])
    #Purpose-> one-hot
    ('purpose',OneHotEncoder(handle_unknown='ignore'),['Purpose']),
], remainder='passthrough')

#Carregar dados -----------------------
df = pd.read_csv("class_german_credit.csv")
y = (df['Risk'] == 'good').astype(int)
X = df.drop(columns='Risk')

# Apply the preprocessor to X
X_processed = preprocessor.fit_transform(X)

# Get names of the resulting features
feature_names = preprocessor.get_feature_names_out()

# Convert to DataFrame
X_processed = pd.DataFrame(
    X_processed,
    columns=feature_names,
    index=X.index
)

# Add Risk back to the processed dataframe
X_processed['Risk'] = y

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy import stats
import seaborn as sns
preprocessor = ColumnTransformer([
    # Sex -> binary
    ('sex',OrdinalEncoder(categories=[['female', 'male']]), ['Sex']),

    # Housing -> ordinal
    ( 'housing',OrdinalEncoder(categories=[['free','rent', 'own']]),['Housing']),

    # Saving accounts
    ('saving',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,categories=[['little','moderate','rich','quite rich']]),['Saving accounts']),
    # Checking account
    ('checking',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,categories=[['little','moderate','rich']]),['Checking account']),
    #Coisas de discretizacao, nao funcionou
    # ('age_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'), ['Age']),

    # ('duration_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'),['Duration']),

    # ('credit_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'),['Credit amount'])
    #Purpose-> one-hot
    ('purpose',OneHotEncoder(handle_unknown='ignore'),['Purpose']),
], remainder='passthrough')

#Carregar dados -----------------------
df = pd.read_csv("class_german_credit.csv")
y = (df['Risk'] == 'good').astype(int)
X = df.drop(columns='Risk')

# Apply the preprocessor to X
X_processed = preprocessor.fit_transform(X)

# Get names of the resulting features
feature_names = preprocessor.get_feature_names_out()

# Convert to DataFrame
X_processed = pd.DataFrame(
    X_processed,
    columns=feature_names,
    index=X.index
)

# Add Risk back to the processed dataframe
X_processed['Risk'] = y

# =========================
# PEARSON
# =========================

resultado_pearson = X_processed.corr(method='pearson')['Risk'].drop('Risk').sort_values()

print('Resultado Pearson =')
print(resultado_pearson)

# Full Pearson matrix
plt.figure(figsize=(16, 13))

sns.heatmap(resultado_pearson,annot=True,fmt=".2f",cmap="coolwarm",center=0,vmin=-1,vmax=1,square=True,linewidths=0.5,cbar_kws={"shrink": 0.8})

plt.title("Matriz de Correlação de Pearson", fontsize=18, pad=20)

plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig("correlation_matrix_pearson.png",dpi=400,bbox_inches="tight")
plt.close()
# =========================
# SPEARMAN
# =========================
resultado_spearman = X_processed.corr(method='spearman')['Risk'].drop('Risk').sort_values()

print('Resultado Spearman =')
print(resultado_spearman)
# Full Spearman matrix
plt.figure(figsize=(16, 13))

sns.heatmap(resultado_spearman,annot=True,fmt=".2f",cmap="coolwarm",center=0,vmin=-1,vmax=1,square=True,linewidths=0.5,cbar_kws={"shrink": 0.8})

plt.title("Matriz de Correlação de Spearman", fontsize=18, pad=20)

plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig("correlation_matrix_spearman.png",dpi=400,bbox_inches="tight")
plt.close()