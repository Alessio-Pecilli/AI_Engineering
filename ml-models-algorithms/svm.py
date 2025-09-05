import matplotlib as plt
import sklearn
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay

RANDOM_SEED = 6

#Maximal margin classifier
X, y = make_blobs(n_samples=40, n_features=2, centers=2, random_state=RANDOM_SEED)
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1e-3)
model.fit(X, y)

C_values = [1e-3, 1, 100]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
models = []
for kernel in kernels:
    for c in C_values:
        svc = SVC(kernel=kernel, C=c)
        svc.fit(X, y)
        models.append(svc)