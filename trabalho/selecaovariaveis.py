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
from scipy.stats import chi2_contingency,chi2
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
numeric_features = [
    'remainder__Credit amount',
    'remainder__Duration',
    "remainder__Job",
    "remainder__Age",
]

categorical_features = [
    "purpose__Purpose_business",
    "purpose__Purpose_vacation/others",
    "purpose__Purpose_car",
    "purpose__Purpose_furniture/equipment",
    "purpose__Purpose_repairs",
    "purpose__Purpose_domestic appliances",
    "purpose__Purpose_education",
    "sex__Sex",
    "purpose__Purpose_radio/TV",
    "housing__Housing",
    "saving__Saving accounts",
    "checking__Checking account",
    "remainder__Job"
]
# Add Risk back to the processed dataframe
X_processed['Risk'] = y

# =========================
# SPEARMAN
# =========================
resultado_spearman = (X_processed.corr(method='spearman')['Risk'].drop('Risk').sort_values(key=abs,ascending=False).to_frame(name='Risk'))
print(resultado_spearman)
plt.figure(figsize=(8, 12))

sns.heatmap(resultado_spearman,annot=True,fmt=".2f",cmap="coolwarm",center=0,vmin=-1,vmax=1,linewidths=0.5,cbar_kws={"label": "Correlação"})

plt.title("Correlação de Spearman com Risk", fontsize=18, pad=20)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("correlation_matrix_spearman.png",dpi=400,bbox_inches="tight")
plt.close()

#QUI Quadrado -----------------------
alpha = 0.05
nomes=[]
for feature in categorical_features:

    tabela = pd.crosstab(
        X_processed[feature],
        X_processed['Risk']
    )

    stat, p, dof, expected = chi2_contingency(tabela)

    # Critical value
    critical = chi2.ppf(1 - alpha, dof)

    if stat >= critical:
        nome=(p,f"""
{feature}
Chi-square = {stat:.4f}
Critical value = {critical:.4f}
p-value = {p:.6f}
Dependent (reject H0)""")
    else:
        print()
        nome=(p,
f"""
{feature}
Chi-square = {stat:.4f}
Critical value = {critical:.4f}
p-value = {p:.6f}
Independent (fail to reject H0)""")
    nomes.append(nome)
nomes.sort()
for i in nomes:
    print(i[1])