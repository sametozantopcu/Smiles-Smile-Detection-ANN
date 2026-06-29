import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import pathlib
dataset_url ="C:/Users/ibrah/PycharmProjects/finger_tracking/SMILEs"
data_dir = pathlib.Path(dataset_url)


batch_size = 50
seq_length = 64
input_size = 64
epochs = 10
num_classes = 2

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_url,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(input_size, seq_length),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(input_size, seq_length),
  batch_size=batch_size)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)


normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(input_size, seq_length,3)),
    tf.keras.layers.Dense(2, activation='sigmoid')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer= optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.build()
model.summary()

history = model.fit(normalized_train_ds, validation_data=normalized_val_ds, epochs=epochs)

# Plot training loss and accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
