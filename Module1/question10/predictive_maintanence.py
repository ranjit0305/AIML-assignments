import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import joblib

df = pd.read_csv("ai4i2020.csv")
print("First 5 Records\n")
print(df.head())

print("\nDataset Information\n")
print(df.info())
print("\nDataset Shape\n")
print(df.shape)
print("\nColumn Names\n")
print(df.columns)
print("\nStatistical Summary\n")
print(df.describe())
print("\nMissing Values\n")
print(df.isnull().sum())
print("\nDuplicate Records :", df.duplicated().sum())

#Failure distribution
print("\nMachine Failure Distribution\n")
print(df["Machine failure"].value_counts())
df["Machine failure"].value_counts().plot(
    kind="bar"
)
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Count")
plt.savefig("target_distribution.png")
plt.show()

#Type distribution
print(df["Type"].value_counts())
df["Type"].value_counts().plot(
    kind="bar"
)
plt.title("Machine Type Distribution")
plt.xlabel("Machine Type")
plt.ylabel("Count")
plt.savefig("machine_type_distribution.png")
plt.show()

# Remove unnecessary columns
df.drop(
    columns=[
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ],
    inplace=True
)
print(df.head())
print(df.columns)

#Encoding
encoder = LabelEncoder()
df["Type"] = encoder.fit_transform(df["Type"])
print(df.head())

#Correlation matrix
plt.figure(figsize=(8,6))
plt.imshow(df.corr(), cmap="Blues")
plt.colorbar()
plt.xticks(
    range(len(df.columns)),
    df.columns,
    rotation=90
)
plt.yticks(
    range(len(df.columns)),
    df.columns
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

#Selecting feautures and target
X = df.drop(
    "Machine failure",
    axis=1
)
y = df["Machine failure"]
print("Feature Shape :", X.shape)
print("Target Shape :", y.shape)

# Train-Test Split

X_train, X_test, y_train, y_tes t = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# Random Forest Model

model = RandomForestClassifier(n_estimators=100,random_state=42,class_weight="balanced")
print("\nTraining Model...\n")
model.fit(X_train,y_train)
print("Model Trained Successfully!")

#Predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"\nAccuracy : {accuracy:.4f}")
print("\nClassification Report\n")
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Error Analysis
false_predictions = X_test.copy()
false_predictions["Actual"] = y_test.values
false_predictions["Predicted"] = y_pred
errors = false_predictions[false_predictions["Actual"] != false_predictions["Predicted"]]
print("\nTotal Misclassified Samples :", len(errors))
print("\nFirst 10 Misclassified Samples\n")
print(errors.head(10))

joblib.dump(
    model,
    "predictive_maintenance_model.pkl"
)
print("\nModel Saved Successfully!")