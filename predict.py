import joblib
import pandas as pd

model = joblib.load("spam_model.pkl")
features = joblib.load("features.pkl")

# Example: fake email vector
email_vector = dict.fromkeys(features, 0)
email_vector["free"] = 2
email_vector["money"] = 1
email_vector["win"] = 1

X_new = pd.DataFrame([email_vector])
prob = model.predict_proba(X_new)[0][1]

print(f"Spam Probability: {prob*100:.2f}%")
