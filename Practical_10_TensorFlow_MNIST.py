# Practical 10: Demo of TensorFlow Tools using MNIST

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize image data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=5)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Predict one image
prediction = model.predict(X_test[:1])
print("Predicted Digit:", prediction.argmax())

# Display test image
plt.imshow(X_test[0], cmap='gray')
plt.title("MNIST Test Image")
plt.axis('off')
plt.show()
