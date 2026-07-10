import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("MNIST Dataset Loaded Successfully!")

print("Training Images Shape:", X_train.shape)
print("Training Labels Shape:", y_train.shape)

print("Testing Images Shape:", X_test.shape)
print("Testing Labels Shape:", y_test.shape)

# Display the first handwritten digit

plt.imshow(X_train[0], cmap="gray")
plt.title(f"Digit: {y_train[0]}")
plt.axis("off")
plt.show()

# Convert 28x28 images into 784 features

X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Normalize pixel values

X_train = X_train / 255.0
X_test = X_test / 255.0

from sklearn.linear_model import LogisticRegression

# Create the Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

print("\nTraining the model...")

# Train the model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Predict on training and testing data

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy : {test_accuracy:.4f}")

# Display 5 Correct Predictions

correct = []

for i in range(len(y_test)):
    if y_test[i] == test_pred[i]:
        correct.append(i)

plt.figure(figsize=(12,3))

for i in range(5):
    plt.subplot(1,5,i+1)

    plt.imshow(X_test[correct[i]].reshape(28,28), cmap="gray")

    plt.title(f"Pred: {test_pred[correct[i]]}")
    plt.axis("off")

plt.suptitle("Correct Predictions")

plt.tight_layout()

plt.savefig("correct_predictions.png")

plt.show()