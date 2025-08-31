# ==============================
# IMPORT LIBRARIES
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, log_loss, recall_score, precision_score,
    f1_score, accuracy_score, classification_report
)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC

# ==============================
# RANDOM SEED
# ==============================
RANDOM_SEED = 0

# ==============================
# BINARY CLASSIFICATION DATASET
# ==============================
X, y = make_classification(
    n_samples=100, n_features=2,
    n_informative=2, n_redundant=0,
    n_classes=2, n_repeated=0,
    random_state=RANDOM_SEED
)

# Visualize dataset
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Binary classification dataset")
plt.show()

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED
)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# ==============================
# DECISION BOUNDARY PLOT
# ==============================
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(model, X, y)

# ==============================
# MODEL EVALUATION (BINARY)
# ==============================
print("Training accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ==============================
# MULTICLASS CLASSIFICATION (OvO vs OvR)
# ==============================
X, y = make_classification(
    n_samples=100, n_features=2,
    n_informative=2, n_redundant=0,
    n_classes=3, n_repeated=0,
    random_state=RANDOM_SEED
)

# Visualize dataset
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Multiclass dataset (3 classes)")
plt.show()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED
)

# One-vs-One
ovo_model = OneVsOneClassifier(SVC())
ovo_model.fit(X_train, y_train)

# One-vs-Rest
ovr_model = OneVsRestClassifier(SVC())
ovr_model.fit(X_train, y_train)

# Accuracy comparison
print("One-vs-One Training accuracy:", ovo_model.score(X_train, y_train))
print("One-vs-One Test accuracy:", ovo_model.score(X_test, y_test))
print("One-vs-Rest Training accuracy:", ovr_model.score(X_train, y_train))
print("One-vs-Rest Test accuracy:", ovr_model.score(X_test, y_test))
