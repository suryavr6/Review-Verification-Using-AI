import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("fake_reviews_dataset.csv")

# Input and output
X = data['text_']
y = data['label']

# Convert text into numbers
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_vector = vectorizer.fit_transform(X)

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

# Save model
pickle.dump(model, open("model/fake_review_model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl", "wb"))

print("Model Saved Successfully")