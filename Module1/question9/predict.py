import cv2
import numpy as np

from tensorflow.keras.models import load_model

model = load_model("face_mask_model.keras")

image_path = "without_mask_1.jpg"      # Replace with your image

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

image = cv2.resize(image, (224, 224))

image = image.astype("float32") / 255.0

image = np.expand_dims(image, axis=0)

prediction = model.predict(image, verbose=0)

probability = prediction[0][0]

print("\nRaw Prediction:", prediction)

if probability >= 0.5:
    print("\nPrediction : WITHOUT MASK")
    print(f"Confidence : {probability:.2%}")
else:
    print("\nPrediction : WITH MASK")
    print(f"Confidence : {(1 - probability):.2%}")