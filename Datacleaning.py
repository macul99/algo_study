import pandas as pd
df = pd.read_csv("data.csv")

# Basic pandas hygiene
df = df.drop_duplicates()
df = df.convert_dtypes()    # get sensible dtypes
# optional: coerce numeric columns
for c in ["age","income"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Column selectors (by dtype)
num_sel  = selector(dtype_include=["number"])
cat_sel  = selector(dtype_include=["object", "string", "category", "bool"])

numeric_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler(with_mean=False))  # with_mean=False handles sparse safely if mixed
])

categorical_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_sel),
        ("cat", categorical_pipe, cat_sel),
    ],
    remainder="drop"
)

model = LogisticRegression(max_iter=200)

clf = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", model)
])

clf.fit(X_train, y_train)
print("Test score:", clf.score(X_test, y_test))

# 3) Adding more “cleaning” (outliers, skew, feature selection)

from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold

numeric_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("yeojohnson", PowerTransformer(method="yeo-johnson")),  # handles skew, nonnegative not required
    ("vt", VarianceThreshold(threshold=0.0)),                 # drop constant numeric features
    ("scale", StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))  # rare levels grouped
])

# 4) Dates or custom fixes with FunctionTransformer

import numpy as np
from sklearn.preprocessing import FunctionTransformer

def add_date_parts(X):
    X = X.copy()
    if "timestamp" in X:
        dt = pd.to_datetime(X["timestamp"], errors="coerce")
        X["year"]  = dt.dt.year
        X["month"] = dt.dt.month
        X["dow"]   = dt.dt.dayofweek
        X = X.drop(columns=["timestamp"])
    return X

date_pipe = FunctionTransformer(add_date_parts, feature_names_out="one-to-one")

clf = Pipeline([
    ("add_dates", date_pipe),
    ("prep", preprocess),
    ("clf", model)
])

# 5) Cross-validation & hyperparameters (no leakage)

from sklearn.model_selection import GridSearchCV

param_grid = {
    "prep__num__impute__strategy": ["median", "mean"],
    "prep__cat__onehot__min_frequency": [None, 5, 10],
    "clf__C": [0.1, 1, 10]
}

gs = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.score(X_test, y_test))
