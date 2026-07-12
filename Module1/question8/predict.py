import cv2
import numpy as np

from tensorflow.keras.models import load_model

# ============================================
# Load Trained Model
# ============================================

model = load_model("model.keras")

# ============================================
# Load Test Image
# ============================================

image_path = "test.png" 

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# ============================================
# Preprocess Image
# ============================================

image = cv2.resize(image, (128, 128))

image = image.astype("float32") / 255.0

image = np.expand_dims(image, axis=0)

# ============================================
# Predict
# ============================================

prediction = model.predict(image, verbose=0)

print("\nRaw Prediction:", prediction)

good_probability = prediction[0][0]
defect_probability = prediction[0][1]

print(f"\nGood Probability      : {good_probability:.4f}")
print(f"Defective Probability : {defect_probability:.4f}")

# ============================================
# Final Prediction
# ============================================

if defect_probability > good_probability:
    print("\nPrediction : DEFECTIVE Screw")
    print(f"Confidence : {defect_probability:.2%}")
else:
    print("\nPrediction : GOOD Screw")
    print(f"Confidence : {good_probability:.2%}")