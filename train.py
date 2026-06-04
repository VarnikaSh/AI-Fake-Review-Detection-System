import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("reviews.csv")

X = df["review"]
y = df["label"]

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression()

model.fit(X_vectorized, y)

pickle.dump(
    model,
    open("model.pkl","wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl","wb")
)

print("Model Saved")