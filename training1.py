from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
import os
import matplotlib as plt
# Required to save models in HDF5 format
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np

print(tf.version.VERSION)

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

classes = ["airplane", "automobile", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck"]


def plot_sample(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

X_train = X_train / 255.0
X_test = X_test / 255.0


# Define a simple sequential model
def create_model():
    cnn = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(5, 5),
                      activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(1000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(500, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(250, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])

    cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return cnn


# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=2)

# Train the model with the new callback
model.fit(X_train, y_train_one_hot, epochs=30, batch_size=256,
            validation_split=0.2,
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.