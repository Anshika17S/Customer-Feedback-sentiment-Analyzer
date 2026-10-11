import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import clean_text

# A. Load dataset
df = pd.read_csv("data/amazon_alexa.tsv", sep="\t")
print("Columns:", list(df.columns))

# Keep only verified_reviews and rating, rename verified_reviews to review
df = df[["verified_reviews", "rating"]].rename(columns={"verified_reviews": "review"})

# B. Basic EDA
print(f"Total rows: {len(df)}")
print(f"Missing reviews: {df['review'].isna().sum()}")
print(f"Duplicate reviews: {df['review'].duplicated().sum()}")

print("Rating counts:")
print(df["rating"].value_counts().sort_index())

word_counts = df["review"].dropna().apply(lambda x: len(str(x).split()))
print(f"Average review length in words: {word_counts.mean():.2f}")

# C. Data cleaning
# Drop rows where review is missing or empty
df = df.dropna(subset=["review"])
df = df[df["review"].astype(str).str.strip() != ""]
print(f"Rows after dropping missing/empty reviews: {len(df)}")

# Drop duplicate reviews
df = df.drop_duplicates(subset=["review"], keep="first")
print(f"Rows after dropping duplicate reviews: {len(df)}")

# Map ratings to labels: 1-2 negative, 3 neutral, 4-5 positive
def map_rating(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df["label"] = df["rating"].apply(map_rating)
print("Label counts:")
print(df["label"].value_counts())
print(f"Rows after labeling: {len(df)}")

# Apply clean_text
df["clean_review"] = df["review"].apply(clean_text)

# Drop rows where clean_review is empty after cleaning
df = df[df["clean_review"] != ""]
print(f"Rows after dropping empty cleaned reviews: {len(df)}")

# D. Train
X = df[["clean_review", "review"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

pipeline.fit(X_train["clean_review"], y_train)

# E. Evaluate and save results
y_pred = pipeline.predict(X_test["clean_review"])
probabilities = pipeline.predict_proba(X_test["clean_review"])

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save results
os.makedirs("results", exist_ok=True)

# Save metrics.csv
report_dict = classification_report(y_test, y_pred, output_dict=True)
metrics_df = pd.DataFrame(report_dict).transpose()
metrics_df.to_csv("results/metrics.csv", index=True)

# Save test_predictions.csv
confidence = probabilities.max(axis=1).round(4)
test_predictions_df = pd.DataFrame({
    "review": X_test["review"].values,
    "actual": y_test.values,
    "predicted": y_pred,
    "confidence": confidence
})
test_predictions_df.to_csv("results/test_predictions.csv", index=False)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/sentiment_model.joblib")
print("Saved model to model/sentiment_model.joblib and results to results/")
