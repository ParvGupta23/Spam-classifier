import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import joblib

# Load data
df = pd.read_csv("data/emails.csv")

# Features & label
X = df.drop(columns=["Email No.", "Prediction"])
y = df["Prediction"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
f1 = f1_score(y_test, preds)

print(f"F1 Score: {f1:.4f}")

# Save model
joblib.dump(model, "spam_model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("Model trained and saved.")
