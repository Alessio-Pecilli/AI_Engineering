import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv("housing.csv")

# Features (X) and target (y)
X = df.drop("price", axis=1)
y = df["price"]

# Identify categorical and numerical columns
categorical_cols = ["furnishingstatus"]
numerical_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]

# ==============================
# PREPROCESSING
# ==============================
num_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ],
    remainder="passthrough"  # keep binary 0/1 columns as they are
)

X_preprocessed = preprocessor.fit_transform(X)

print("Shape before preprocessing:", X.shape)
print("Shape after preprocessing:", X_preprocessed.shape)

# ==============================
# MODELS
# ==============================
lr = LinearRegression()
l = Lasso(alpha=0.1, max_iter=1000)
en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
ridge_model = Ridge(alpha=0.1, max_iter=1000)

models = {
    "LinearRegression": lr,
    "Lasso": l,
    "ElasticNet": en,
    "Ridge": ridge_model
}

# Fit all models
for name, model in models.items():
    model.fit(X_preprocessed, y)

# ==============================
# PRINT RESULTS
# ==============================
print("Lasso coefficients:", l.coef_)
print("ElasticNet coefficients:", en.coef_)
print("Ridge coefficients:", ridge_model.coef_)

print("Number of zero coefficients (Lasso):", np.sum(l.coef_ == 0))
print("Number of zero coefficients (ElasticNet):", np.sum(en.coef_ == 0))

# Cross-validation on Linear Regression
cv_results = cross_validate(lr, X_preprocessed, y, cv=5, scoring="r2", return_train_score=True)
print("Cross-Validation Train R2 (Linear Regression):", cv_results['train_score'])

# ==============================
# VISUALIZATION
# ==============================
# Barplot of R2 scores
plt.figure(figsize=(10, 6))
r2_scores = [model.score(X_preprocessed, y) for model in models.values()]
sns.barplot(x=list(models.keys()), y=r2_scores)
plt.ylabel("R2 Score")
plt.title("Comparison of R2 scores across models")
plt.show()

# Residuals for Linear Regression
y_pred_lr = lr.predict(X_preprocessed)
resid_lr = y - y_pred_lr
plt.figure(figsize=(8, 5))
sns.histplot(resid_lr, kde=True)
plt.xlabel("Residuals")
plt.title("Residual distribution (Linear Regression)")
plt.show()
