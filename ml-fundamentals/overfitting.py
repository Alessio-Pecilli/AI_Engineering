# ==============================
# IMPORT LIBRARIES
# ==============================
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_validate


# ==============================
# DATASET LOADING
# ==============================
RANDOM_SEED = 0
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target

# Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED
)


# ==============================
# PREPROCESSING
# ==============================
# Standardize features (important for regularization methods)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Add polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# ==============================
# MODELS
# ==============================

# 1) Linear Regression (no regularization)
lr = LinearRegression()
lr.fit(X_train_poly, y_train)
y_pred = lr.predict(X_test_poly)
print("Linear Regression R2:", r2_score(y_test, y_pred))

# 2) ElasticNet (combination of Lasso and Ridge)
en = ElasticNet(alpha=500)   # stronger penalty
en.fit(X_train_poly, y_train)
y_pred = en.predict(X_test_poly)
print("ElasticNet R2:", r2_score(y_test, y_pred))

# 3) Cross-validation with ElasticNet
print(cross_validate(en, X, y, cv=5, scoring='r2', return_train_score=True))
