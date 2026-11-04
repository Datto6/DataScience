import pandas as pd
import numpy as np

from sklearn import tree  # Arvore de decisão e plot tree
from sklearn.metrics import accuracy_score   # Acurácia
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, MinMaxScaler,KBinsDiscretizer,FunctionTransformer  # Transformar coluna ordinária
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold  # Separar a parte de teste
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,f1_score  # Matriz de confusão
import matplotlib.pyplot as plt  # Plot na tabela
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# =========================
# LOAD DATA
# =========================

df = pd.read_csv("class_german_credit.csv")
# df=df.dropna().reset_index(drop=True)
# =========================
# COLUMN GROUPS
# =========================

binary_cols = ['Sex']
housing_col = ['Housing']
saving_col = ['Saving accounts']
checking_col = ['Checking account']

# =========================
# PREPROCESSING PIPELINE
# =========================

# target
y = (df['Risk'] == 'good').astype(int)
# print(df.head()) so para olhar valores
# features
X = df.drop(columns='Risk')

preprocessor = ColumnTransformer([
    # Sex -> binary
    ('sex',OrdinalEncoder(categories=[['female', 'male']]), binary_cols),

    # Housing -> ordinal
    ( 'housing',OrdinalEncoder(categories=[['free','rent', 'own']]),housing_col),

    # Saving accounts
    ('saving',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,categories=[['little','moderate','rich','quite rich']]),saving_col),
    # Checking account
    ('checking',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,categories=[['little','moderate','rich']]),checking_col),
    #Coisas de discretizacao, nao funcionou
    # ('age_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'), ['Age']),

    # ('duration_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'),['Duration']),

    # ('credit_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'),['Credit amount'])
    #Purpose-> one-hot
], remainder='passthrough')

# ============================================================
# ENCODE PURPOSE ONLY
# ============================================================

purpose_encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
purpose_encoded = purpose_encoder.fit_transform(X[['Purpose']])
purpose_columns = purpose_encoder.get_feature_names_out(['Purpose'])
purpose_df = pd.DataFrame(purpose_encoded,columns=purpose_columns,index=X.index)

# Remove original Purpose
X = X.drop(columns=['Purpose'])

# Add one-hot Purpose columns
X = pd.concat([X, purpose_df],axis=1)

# =========================
# COMPLETE PIPELINE
# =========================

pipe = Pipeline([
    ('preprocessing', preprocessor), #transforma em numericos 
    ('imputer', KNNImputer(n_neighbors=5)),
    ('normalizer',MinMaxScaler()), #normaliza
    ('model', KNeighborsClassifier()),
])

# =========================
# TRAIN TEST SPLIT
# =========================
worst_attributes = [
    'Purpose_business',
    'Purpose_car',
    'Purpose_furniture/equipment',
    'Purpose_repairs',
    'Purpose_domestic appliances',
    'Purpose_education'
]

X = X.drop(columns=worst_attributes)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Fazendo cross validation  para achar melhores hiperparametros

param_grid = {
    'model__n_neighbors': [3, 5, 7, 9, 11],
    'model__weights': ['uniform', 'distance']
}
 
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='accuracy'
)
#Treina no espaço de treinamento
grid.fit(X_train, y_train)
#Apresentar resultados
best_model = grid.best_estimator_
print(f"Best parameters: {grid.best_params_}")

y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
f1_good = f1_score(y_test, y_pred, pos_label=1)
f1_bad = f1_score(y_test, y_pred, pos_label=0)
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy:       {test_accuracy*100:.2f}%")
print(f"Good credit F1: {f1_good*100:.2f}%")
print(f"Bad credit F1:  {f1_bad*100:.2f}%")
print(f"Macro F1:       {f1_macro*100:.2f}%")


y_pred = best_model.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"{acuracia*100:.2f}% best model accuracy")

best_cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=best_cm)
disp.plot(cmap='Blues')
plt.title('Best Confusion Matrix')
plt.show()