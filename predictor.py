import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier



df = pd.read_csv("heart_disease_uci.csv")


#Data preprocessing

df = df.drop(columns=["ca", "thal", "slope"]) #to many missing values
df = df.drop(columns=["id","dataset"]) #irrelevant to train

numeric_columns=["age","trestbps","chol","thalch","oldpeak"]
cat_columns=["fbs","exang","restecg","sex","cp"]

df["num"] = (df["num"] > 0).astype(int) 
df = df.replace(r'^\s*$', np.nan, regex=True)

X = df.drop(columns=["num"])
y = df["num"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Filling in Missing values

num_imputer = SimpleImputer(strategy="median")
X_train[numeric_columns] = num_imputer.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = num_imputer.transform(X_test[numeric_columns])


cat_imputer = SimpleImputer(strategy="most_frequent")
X_train[cat_columns] = cat_imputer.fit_transform(X_train[cat_columns])
X_test[cat_columns] = cat_imputer.transform(X_test[cat_columns])


X_train = pd.get_dummies(X_train, columns=cat_columns, dtype=int)
X_test = pd.get_dummies(X_test, columns=cat_columns, dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)



#-----------------------------------------------------------------

#Standardizing Numeric Columns

scaler = StandardScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

#-----------------------------------------------------

#Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print(classification_report(y_test, y_pred))

#------------------------------------------------------

#Grid Search Decision Tree

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Filling in Missing values

num_imputer = SimpleImputer(strategy="median")
X_train[numeric_columns] = num_imputer.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = num_imputer.transform(X_test[numeric_columns])


cat_imputer = SimpleImputer(strategy="most_frequent")
X_train[cat_columns] = cat_imputer.fit_transform(X_train[cat_columns])
X_test[cat_columns] = cat_imputer.transform(X_test[cat_columns])


X_train = pd.get_dummies(X_train, columns=cat_columns, dtype=int)
X_test = pd.get_dummies(X_test, columns=cat_columns, dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

param_grid = {
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_leaf": [1, 5, 10, 20],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy", 
)
grid.fit(X_train, y_train)
best_tree = grid.best_estimator_
print(classification_report(y_test, best_tree.predict(X_test)))


#-------------------------------------------------------------

#GradientBoosting

df = pd.read_csv("heart_disease_uci.csv")

df = df.drop(columns=["id","dataset"]) #irrelevant to train

df["num"] = (df["num"] > 0).astype(int) 
df = df.replace(r'^\s*$', np.nan, regex=True)

X = df.drop(columns=["num"])
y = df["num"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

cat_columns = ["fbs", "exang", "restecg", "sex", "cp", "thal", "slope"]

X_train = pd.get_dummies(X_train, columns=cat_columns, dtype=int)
X_test = pd.get_dummies(X_test, columns=cat_columns, dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

hgb = HistGradientBoostingClassifier(random_state=42)

param_grid = {
    "learning_rate": [0.03, 0.1],
    "max_depth": [2, 3, None],
    "max_iter": [100, 300],
    "min_samples_leaf": [10, 20],
}
grid = GridSearchCV(hgb, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test) 
print(classification_report(y_test, y_pred))



