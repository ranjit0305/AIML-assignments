import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

df = pd.read_csv("Titanic-Dataset.csv")
df.info()
print(df.head())

print("\nDataset Information\n")
print(df.info())

print("\nMissing Values\n")
print(df.isnull().sum())

print(df["Survived"].value_counts())
df["Survived"].value_counts().plot(kind="bar")

plt.title("Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Count")

plt.savefig("survival_distribution.png")

plt.show()

plt.figure(figsize=(8,5))

df["Age"].hist(bins=20)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.savefig("age_distribution.png")

plt.show()

print(df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop(columns=["Cabin"], inplace=True)

# Convert Sex column into numeric values
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# Convert Embarked column into numeric values
df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

print("\nDataset after Feature Engineering:\n")
print(df.head())

# Features
X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
# Target
y = df["Survived"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))
# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
print("\nLogistic Regression trained successfully!")
log_predictions = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_predictions)
print(f"\nLogistic Regression Accuracy: {log_accuracy:.4f}")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest trained successfully!")
# Predictions
rf_predictions = rf_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")

print("\n----------- Model Comparison -----------")

print(f"Logistic Regression Accuracy : {log_accuracy:.4f}")
print(f"Random Forest Accuracy       : {rf_accuracy:.4f}")

if rf_accuracy > log_accuracy:
    print("\nRecommended Model: Random Forest")
else:
    print("\nRecommended Model: Logistic Regression")