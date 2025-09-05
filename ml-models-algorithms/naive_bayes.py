from sklearn.datasets import load_iris, make_gaussian_quantiles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, classification_report, confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

RANDOM_SEED = 0

#Gaussian Naive Bayes, gaussian distribution
X, y = make_gaussian_quantiles(n_features=1, n_classes=2, random_state=RANDOM_SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_test, y_test)

y_prova = gnb.predict_proba(X_test)
log_loss(y_test, y_prova)
print("Valori X_test:")
print(X_test)
print("Valori y_test:")
print(y_test)
print("Valori predetti da GaussianNB:")
print(gnb.predict(X_test))
print("Probabilità predette da GaussianNB:")
print(y_prova)

#Bernoulli Naive Bayes, bernoulli distribution
df = pd.read_csv("spam.csv")

def build_dictionary(corpus):
    words = []
    for doc in corpus:
        for word in doc.split(" "):
            if word not in words:
                words.append(word.lower())    
    return words

def build_efficient_vocab(corpus):
    vocab = set()
    for doc in corpus:
        vocab.update(set(doc.lower().split(" ")))
    return list(vocab)

def binary_bow(corpus, vocab = None):
    if vocab is None:
        vocab = build_efficient_vocab(corpus)
    
    n = len(corpus)
    m = len(vocab)
    
    X = np.zeros((n, m))
    
    for i, doc in enumerate(corpus):
        for word in doc.split(" "):
            if word in vocab:
                j = vocab.index(word)
                X[i, j] = 1
    return X, vocab

sms_list = df["MESSAGE"].tolist()
X = binary_bow(sms_list)
y = df["SPAM"].to_list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

from sklearn.naive_bayes import BernoulliNB

benb = BernoulliNB()
benb.fit(X_train, y_train)
benb.score(X_test, y_test)

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnb.score(X_test, y_test)

from sklearn.naive_bayes import ComplementNB

cnb = ComplementNB()
cnb.fit(X_train, y_train)
cnb.score(X_test, y_test)

