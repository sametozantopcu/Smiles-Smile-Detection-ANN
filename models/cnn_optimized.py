import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# Local dataset directory
data_directory = pathlib.Path("C:/Users/ibrah/PycharmProjects/finger_tracking/SMILEs")

# Global variables
batch_size = 50
img_height = 64
img_width = 64
epochs = 10

# Load training split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Load validation split
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Normalize grayscale pixel intensity values from [0, 255] to [0, 1]
rescale_layer = layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda img, lbl: (rescale_layer(img), lbl))
val_ds = val_ds.map(lambda img, lbl: (rescale_layer(img), lbl))

# Setup dataset pipelines using tf.data to prevent loading latency
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Multilayer Perceptron Network (FFNN)
# Removed the intermediate softmax layer from previous trial because it broke the backpropagation gradient flow.
# Kept standard relu for dense layers and switched to 2-node output with SparseCategoricalCrossentropy.
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(img_height, img_width, 3)),
    layers.Dense(128, activation='relu'),
    layers.Dense(150, activation='relu'),
    layers.Dense(2) 
])

# Changed optimizer config to Adam with custom learning rate
# Fixed logit parameter conflict in the loss calculation to stabilize convergence
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=custom_optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Check model parameters
model.summary()

# Training loop
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=epochs
)

# Plotting metrics for project report validation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.title('Loss Convergence')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.title('Accuracy Tracking')
plt.legend()
plt.show()
