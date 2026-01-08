import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv("data/emails.csv")

X = df.drop(columns=["Email No.", "Prediction"])
y = df["Prediction"]

model = joblib.load("spam_model.pkl")
preds = model.predict(X)

print(classification_report(y, preds))
