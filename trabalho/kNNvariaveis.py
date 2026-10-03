import pandas as pd
import numpy as np

from itertools import combinations

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv("class_german_credit.csv")

# ============================================================
# TARGET
# ============================================================

y = (df['Risk'] == 'good').astype(int)
X = df.drop(columns='Risk')

# ============================================================
# ENCODE PURPOSE ONLY
# ============================================================

purpose_encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)

purpose_encoded = purpose_encoder.fit_transform(
    X[['Purpose']]
)

purpose_columns = purpose_encoder.get_feature_names_out(
    ['Purpose']
)

purpose_df = pd.DataFrame(
    purpose_encoded,
    columns=purpose_columns,
    index=X.index
)

# Remove original Purpose
X = X.drop(columns=['Purpose'])

# Add one-hot Purpose columns
X = pd.concat(
    [X, purpose_df],
    axis=1
)


# ============================================================
# 8 WORST ATTRIBUTES
# ============================================================

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


# ============================================================
# CHECK THAT ALL ATTRIBUTES EXIST
# ============================================================

print("Attributes being tested:")

for attribute in worst_attributes:
    if attribute in X.columns:
        print(f"  OK: {attribute}")
    else:
        print(f"  NOT FOUND: {attribute}")


# ============================================================
# RESULTS
# ============================================================

results = []

# ============================================================
# BRUTE FORCE
# ============================================================

total_combinations = 2 ** len(worst_attributes)

for mask in range(total_combinations):
    # Determine which attributes to remove
    removed = []

    for i, attribute in enumerate(worst_attributes):
        if mask & (1 << i): #cada mask é um 10101010 representando levar ou nao levar, cada i checa quais vamos retirar(1 é retirar)
            removed.append(attribute)

    # Create dataset for this combination
    X_temp = X.drop(columns=removed,errors='ignore')

    # Determine categorical columns that still exist
    transformers = []
    # Sex
    if 'Sex' in X_temp.columns:
        transformers.append(('sex',OrdinalEncoder(categories=[['female', 'male']]),['Sex']))


    # Housing
    if 'Housing' in X_temp.columns:
        transformers.append(('housing',OrdinalEncoder(categories=[['free', 'rent', 'own']]),['Housing']))


    # Saving accounts
    if 'Saving accounts' in X_temp.columns:
        transformers.append(
            ('saving',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,
                categories=[['little', 'moderate', 'rich', 'quite rich']]),['Saving accounts']))


    # Checking account
    if 'Checking account' in X_temp.columns:
        transformers.append(
            ('checking',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,categories=[['little', 'moderate', 'rich']]),
            ['Checking account']))


    # --------------------------------------------------------
    # Purpose dummy columns are already numeric
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(transformers,remainder='passthrough')


    # ========================================================
    # PIPELINE
    # ========================================================

    pipe = Pipeline([
        ('preprocessing', preprocessor),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('normalizer', MinMaxScaler()),
        ('model', KNeighborsClassifier())
    ])
    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X_temp,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # ========================================================
    # GRID SEARCH
    # ========================================================

    param_grid = {
        'model__n_neighbors': [3, 5, 7, 9, 11],

        'model__weights': [
            'uniform',
            'distance'
        ]
    }


    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    # ========================================================
    # TRAIN
    # ========================================================
    grid.fit(X_train,y_train)
    # ========================================================
    # TEST
    # ========================================================
    y_pred = grid.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(y_test,y_pred)

    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({
        'removed': ', '.join(removed)if removed else 'None',
        'num_removed': len(removed),
        'cv_accuracy': grid.best_score_,
        'test_accuracy': test_accuracy,
        'best_neighbors': grid.best_params_['model__n_neighbors'],
        'best_weights': grid.best_params_['model__weights']
    })

    # ========================================================
    # PROGRESS
    # ========================================================

    print(
        f"[{mask + 1:3}/{total_combinations}] "
        f"Removed: {removed} | "
        f"CV: {grid.best_score_ * 100:.2f}% | "
        f"Test: {test_accuracy * 100:.2f}%"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

# ============================================================
# SORT BY TEST ACCURACY
# ============================================================

results_df = results_df.sort_values(by='test_accuracy',ascending=False)
# ============================================================
# PRINT TOP 20
# ============================================================

print("\n")
print("=" * 70)
print("TOP 20 COMBINATIONS")
print("=" * 70)

print(
    results_df.head(20).to_string(
        index=False
    )
)
# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "bruteforce_worst_attributes.csv",
    index=False
)

print("\nResults saved to:")
print("bruteforce_worst_attributes.csv")