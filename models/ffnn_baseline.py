import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pathlib

data_directory = pathlib.Path("C:/Users/ibrah/PycharmProjects/finger_tracking/SMILEs")

batch_size = 50
img_height = 64
img_width = 64
epochs = 10

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

rescale_layer = layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda img, lbl: (rescale_layer(img), lbl))
val_ds = val_ds.map(lambda img, lbl: (rescale_layer(img), lbl))

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
    layers.Flatten(input_shape=(img_height, img_width, 3)),
    layers.Dense(2)
])

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=custom_optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=epochs
)

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
