# ==============================
# IMPORT LIBRARIES
# ==============================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing


# ==============================
# LINEAR REGRESSION CLASS (from scratch)
# ==============================
class LinearRegression:
    coef_ = None        # Model coefficients (slopes)
    intercept_ = None   # Intercept (bias)

    def fit(self, X, y):
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Add a column of ones for the intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Solve using the pseudoinverse (numerically stable)
        theta = np.linalg.pinv(X_b).dot(y)

        # Store parameters
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def predict(self, X):
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept_, self.coef_])


# ==============================
# EVALUATION METRICS
# ==============================
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error_custom(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error_custom(y_true, y_pred))

def rss(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)

def sst(y_true):
    return np.sum((y_true - np.mean(y_true))**2)

def r2_score(y_true, y_pred):
    return 1 - rss(y_true, y_pred) / sst(y_true)


# ==============================
# EXAMPLE: CALIFORNIA HOUSING
# ==============================
housing = fetch_california_housing(as_frame=True)
X_train, y_train = housing.data, housing.target

# Convert to numpy arrays
x_train = X_train.to_numpy()
y_train = y_train.to_numpy()

# Fit custom Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_train)

# Print evaluation metrics
print("MAE:", mean_absolute_error(y_train, y_pred))
print("R2:", r2_score(y_train, y_pred))
