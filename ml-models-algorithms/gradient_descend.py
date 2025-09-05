import numpy as np
import sklearn
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import SGDRegressor, SGDClassifier

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    print("MSE: %.3f" % mse)
    print("RMSE: %.3f" % np.sqrt(mse))
    print("R^2: %.3f" % model.score(X_test, y_test))

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

model = SGDRegressor(max_iter=100)
model.fit(X_train, y_train)
evaluate_model(model, X_test, y_test)

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, random_state=42)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_test == y_pred)
    print("Accuracy: %.3f" % accuracy)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)
model = SGDClassifier(max_iter=100)
model.fit(X_train, y_train)
evaluate_model(model, X_test, y_test)

