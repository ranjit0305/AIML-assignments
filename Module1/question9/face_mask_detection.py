import os
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout
)

from tensorflow.keras.models import Model

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)
validation_data = datagen.flow_from_directory(
    "dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)
base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224,224,3)

)
base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

predictions = Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)
model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

model.summary()

print("\nTraining Model...\n")

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=5

)
print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(validation_data)

print(f"\nValidation Accuracy : {accuracy:.4f}")
import numpy as np

predictions = model.predict(validation_data)

y_pred = (predictions > 0.5).astype(int)

y_true = validation_data.classes

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

print(f"\nPrecision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print("\nClassification Report\n")

print(classification_report(y_true, y_pred))
sample_image = next(iter(validation_data))[0][0:1]

start = time.time()

model.predict(sample_image)

end = time.time()

print(f"\nInference Time : {(end-start)*1000:.2f} ms")
model.save("face_mask_model.keras")

print("\nModel Saved Successfully!")

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

plt.savefig("face_mask_accuracy.png")

plt.show()