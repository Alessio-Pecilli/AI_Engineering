import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logging

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.layers import InputLayer, Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ------------------------------
# Example 1: Basic Sequential Model
# ------------------------------
def get_seq_model():
    """Return a Sequential model with 2 hidden layers and softmax output"""
    seq_model = Sequential()
    seq_model.add(InputLayer(input_shape=(100,)))
    seq_model.add(Dense(128, activation='relu'))
    seq_model.add(Dense(32, activation='relu'))
    seq_model.add(Dense(10, activation='softmax'))
    return seq_model

clear_session()
print(get_seq_model().summary())

# Forward propagation example
x = np.random.rand(5, 100)
model = get_seq_model()
y = model.predict(x)
print("Predictions (rounded):")
print(np.round(y, 2))
print("Row sums (should be close to 1):")
print(np.round(np.sum(y, axis=1), 4))


# ------------------------------
# Example 2: Custom activation function
# ------------------------------
def get_seq_activation(activation_name):
    """Return Sequential model with a custom activation function"""
    seq_model = Sequential()
    seq_model.add(InputLayer(input_shape=(100,)))
    seq_model.add(Dense(128, activation='relu'))
    seq_model.add(Dense(32, activation='relu'))
    seq_model.add(Dense(5, activation=activation_name))
    return seq_model

def custom_activation(x):
    return x * 1000   # Just an example

model = get_seq_activation(custom_activation)
y = model.predict(x)


# ------------------------------
# Example 3: Iris dataset model
# ------------------------------
def get_iris_net():
    """Return a Sequential model for Iris classification"""
    iris_model = Sequential()
    iris_model.add(InputLayer(input_shape=(4,)))
    iris_model.add(Dense(32, activation='relu'))
    iris_model.add(Dense(32, activation='relu'))
    iris_model.add(Dense(3, activation='softmax'))
    return iris_model

data = load_iris()
X, y = data['data'], data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clear_session()
iris_model = get_iris_net()
iris_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

print("Training Iris model...")
iris_model.fit(X_train, y_train, epochs=10, verbose=0)
print("Evaluating Iris model...")
print(iris_model.evaluate(X_test, y_test))


# ------------------------------
# Example 4: MNIST dataset (Sequential model)
# ------------------------------
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

plt.matshow(X_train[0], cmap='gray')  # Display an example digit
plt.show()

# Flatten images
x_train_flat = X_train.reshape(-1, 28 * 28).astype('float32') / 255
x_test_flat = X_test.reshape(-1, 28 * 28).astype('float32') / 255

def get_mnist_model():
    """Return Sequential model for MNIST classification"""
    mnist_model = Sequential()
    mnist_model.add(InputLayer(input_shape=(28*28,)))
    mnist_model.add(Dense(512, activation='relu'))
    mnist_model.add(Dense(128, activation='relu'))
    mnist_model.add(Dense(10, activation='softmax'))
    return mnist_model

clear_session()
mnist_model = get_mnist_model()
print(mnist_model.summary())

mnist_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mnist_model.fit(x_train_flat, y_train, epochs=5, batch_size=32, verbose=1)

print("Evaluating MNIST model...")
print(mnist_model.evaluate(x_test_flat, y_test))

# Plot training history
history = mnist_model.fit(
    x_train_flat, y_train,
    epochs=5, batch_size=32,
    validation_data=(x_test_flat, y_test)
)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# ------------------------------
# Example 5: Functional API for MNIST (binary classification even/odd)
# ------------------------------
# Transform labels into even (0) vs odd (1)
y_train_bin = (y_train % 2 != 0).astype(int)
y_test_bin = (y_test % 2 != 0).astype(int)

def functional_model_mnist():
    """Return Functional API model for even/odd MNIST classification"""
    inputs = Input(shape=(28*28,))
    x = Dense(512, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

clear_session()
func_model = functional_model_mnist()
func_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

func_model.fit(x_train_flat, y_train_bin, epochs=5, batch_size=32, verbose=1)
print("Evaluating Functional model...")
print(func_model.evaluate(x_test_flat, y_test_bin))
