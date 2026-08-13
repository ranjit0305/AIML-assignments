import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load Dataset

dataset_path = "screw"

images = []
labels = []


# Load GOOD training images


train_good = os.path.join(dataset_path, "train", "good")

for image_name in os.listdir(train_good):

    image_path = os.path.join(train_good, image_name)

    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.resize(image, (128, 128))

    image = image.astype("float32") / 255.0

    images.append(image)

    labels.append(0)          # Good

# Load Test Images

test_path = os.path.join(dataset_path, "test")

for folder in os.listdir(test_path):

    folder_path = os.path.join(test_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(image, (128, 128))

        image = image.astype("float32") / 255.0

        images.append(image)

        if folder == "good":
            labels.append(0)
        else:
            labels.append(1)

# Convert to NumPy Arrays

images = np.array(images)

labels = np.array(labels)

print("Total Images :", len(images))
print("Image Shape  :", images.shape)
print("Labels Shape :", labels.shape)

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("\nTraining Images :", len(X_train))
print("Testing Images  :", len(X_test))

# One-Hot Encoding

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Data Augmentation

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# Build CNN Model

model = Sequential([

    Input(shape=(128,128,3)),

    Conv2D(
        32,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D((2,2)),

    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D((2,2)),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        2,
        activation="softmax"
    )

])

# Compile Model

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nModel Summary\n")

model.summary()

# Train Model

history = model.fit(

    datagen.flow(
        X_train,
        y_train,
        batch_size=32
    ),

    epochs=10,

    validation_data=(X_test, y_test)

)

# Evaluate

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f"\nTest Accuracy : {accuracy:.4f}")

# Save Model
model.save("model.keras")

print("\nModel saved successfully!")

# Plot Accuracy

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("accuracy_plot.png")

plt.show()