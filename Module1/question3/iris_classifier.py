from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target
print("Features:")
print(iris.feature_names)

print("\nTarget Classes:")
print(iris.target_names)

print("\nTotal Samples:", len(X))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# Predict the test data
y_pred = model.predict(X_test)
print("\nPredicted Values:")
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test,y_pred,target_names=iris.target_names))

plt.figure(figsize=(15,10))

tree.plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("Decision Tree for Iris Classification")
plt.savefig("decision_tree.png")
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap="Blues") # to show as image
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1,2], iris.target_names)
plt.yticks([0,1,2], iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(
            j,
            i,
            cm[i][j],
            ha="center",
            va="center",
            color="black"
        )
plt.savefig("confusion_matrix.png")
plt.show()