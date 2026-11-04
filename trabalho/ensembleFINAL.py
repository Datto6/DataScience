import pandas as pd
import numpy as np

from sklearn import tree  # Arvore de decisão e plot tree
from sklearn.metrics import accuracy_score,f1_score   # Acurácia
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, MinMaxScaler,KBinsDiscretizer,FunctionTransformer  # Transformar coluna ordinária
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold  # Separar a parte de teste
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Matriz de confusão
import matplotlib.pyplot as plt  # Plot na tabela
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
# =========================
# LOAD DATA
# =========================

df = pd.read_csv("class_german_credit.csv")

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
    ('preprocessing', preprocessor), #transforma em numericos p arvore
    ('imputer', KNNImputer(n_neighbors=5)), #imputa missing de saving e checking
    # ('rounder',FunctionTransformer(np.round)), #agora arredonda
    ('model', GradientBoostingClassifier(random_state=42,max_depth=2,n_estimators=100))
])

# =========================
# TRAIN TEST SPLIT
# =========================

worst_attributes = [
    'Job',
    'Purpose_business',
    'Purpose_car',
    'Purpose_furniture/equipment',
    'Purpose_vacation/others',
    'Purpose_repairs',
    'Purpose_domestic appliances',
    'Purpose_education'
]

# Remove only columns that exist
X = X.drop(columns=worst_attributes)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# =========================
# MANUAL MODEL PARAMETERS
# =========================

pipe.set_params(
    model__max_depth=2,
    model__n_estimators=100
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)

f1_good = f1_score(y_test, y_pred, pos_label=1)
f1_bad = f1_score(y_test, y_pred, pos_label=0)
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy:       {acuracia*100:.2f}%")
print(f"Good credit F1: {f1_good*100:.2f}%")
print(f"Bad credit F1:  {f1_bad*100:.2f}%")
print(f"Macro F1:       {f1_macro*100:.2f}%")

# =========================
# CONFUSION MATRIX
# =========================

best_cm = confusion_matrix(y_test,y_pred,labels=[0, 1])

disp = ConfusionMatrixDisplay(confusion_matrix=best_cm,display_labels=['Bad credit', 'Good credit'])
disp.plot(cmap='Blues')
plt.title('Gradient Boosting Confusion Matrix')
plt.show()