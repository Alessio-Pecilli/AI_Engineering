import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from viz import plot_decision_boundary, plot_2d_scatter

RANDOM_SEED = 0

from sklearn.datasets import make_moons, make_circles, make_classification

X, y = make_moons(n_samples=100, noise=0.25, random_state=RANDOM_SEED, shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED)

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

from sklearn.linear_model import Perceptron
model = Perceptron(random_state=RANDOM_SEED)
model.fit(X_train, y_train)
model.score(X_test, y_test)

plot_decision_boundary(model, X, y)


from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=1000, random_state=RANDOM_SEED)
model.fit(X_train, y_train)
model.score(X_test, y_test)

