import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)
print(df.head())
print("\nDataset Information\n")
print(df.info())

df = df[["v1", "v2"]]

df.columns = ["Label", "Message"]

print(df.head())
print(df.isnull().sum())
df.dropna(inplace=True)
df["Label"] = df["Label"].map({
    "ham": 0,
    "spam": 1
})

print(df.head())
X = df["Message"]
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

print(X_train.shape)
# Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)
print("\nModel trained successfully!")
# Predictions
y_pred = model.predict(X_test)
print("\nPredictions Completed!")
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print(f"\nPrecision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("\nClassification Report\n")

print(classification_report(y_test, y_pred))