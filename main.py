import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("fake_reviews_dataset.csv")

# =========================
# INPUT AND OUTPUT
# =========================

X = data['text_']
y = data['label']

# =========================
# TF-IDF VECTORIZATION
# =========================

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_vector = vectorizer.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL
# =========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# =========================
# ACCURACY
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

# =========================
# SAVE MODEL
# =========================

pickle.dump(model, open("model/fake_review_model.pkl", "wb"))

pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl", "wb"))

print("Model Saved Successfully")