# ============================================================
#  Sentiment Analysis - Model Training Script
#  Technique: NLP + Logistic Regression (TF-IDF)
#  BE AIML VI Semester | Cloud Computing Assignment
#  Course Instructor: Dr. Humera Shaziya
# ============================================================

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("   Sentiment Analysis - Model Training")
print("   Technique: NLP + TF-IDF + Logistic Regression")
print("=" * 60)

# ── Dataset ──
positive_texts = [
    "I absolutely love this product it is amazing",
    "This is the best thing I ever bought",
    "Fantastic quality and very fast delivery",
    "Highly recommended excellent experience",
    "I am so happy with this purchase",
    "The service was outstanding and friendly",
    "Great value for money very satisfied",
    "Wonderful product works perfectly",
    "Very pleased with the quality",
    "Superb performance totally worth it",
    "Incredible results I am thrilled",
    "Love it will buy again for sure",
    "Excellent customer support very helpful",
    "Top notch product highly satisfied",
    "Perfect exactly what I was looking for",
    "Amazing experience from start to finish",
    "Very good quality and fast shipping",
    "Best purchase I made this year",
    "Absolutely brilliant exceeded expectations",
    "Five stars would definitely recommend",
    "So glad I bought this works great",
    "Outstanding quality very impressed",
    "Really happy with this great product",
    "Loved the product will order again",
    "Terrific quality arrived quickly",
    "Beautiful design and great functionality",
    "This made my day so wonderful",
    "Impressive features very easy to use",
    "Delighted with the outcome great job",
    "Worth every penny outstanding product",
    "Super fast and reliable great service",
    "Fantastic I love everything about it",
    "Beyond my expectations truly amazing",
    "Happily surprised by the quality",
    "I feel so positive about this purchase",
]

negative_texts = [
    "This product is terrible and useless",
    "Very disappointed with the quality",
    "Worst purchase I have ever made",
    "Absolutely horrible do not buy this",
    "Complete waste of money very bad",
    "Broke after one day very poor quality",
    "Terrible customer service very rude staff",
    "Product did not work at all frustrated",
    "Very unhappy with this purchase",
    "Awful experience would not recommend",
    "Defective product very disappointing",
    "Total junk threw it away immediately",
    "Horrible quality regret buying this",
    "Not worth the price at all",
    "Extremely poor packaging and quality",
    "Delayed delivery and damaged product",
    "Very bad experience poor support",
    "Cheap material broke instantly",
    "Does not match the description at all",
    "Worst experience ever avoid this",
    "Terrible smell low quality material",
    "Does not work as advertised big scam",
    "Very flimsy and broke on first use",
    "Waste of time and money",
    "Extremely disappointed expected better",
    "Never buying from this store again",
    "Absolutely useless product returned it",
    "Broke within hours of opening",
    "Customer service was no help at all",
    "Regret this purchase entirely",
    "Faulty product arrived in bad condition",
    "Complete disaster avoid at all costs",
    "Disappointed by the poor craftsmanship",
    "Terrible experience from start to finish",
    "This is the worst product I have used",
]

neutral_texts = [
    "The product is okay nothing special",
    "It works as described average quality",
    "Delivery was on time product is fine",
    "Neither good nor bad just average",
    "The item is decent for the price",
    "It is what it is nothing extraordinary",
    "Product is acceptable meets basic needs",
    "Average performance could be better",
    "Mediocre quality but functional",
    "It does the job nothing more",
    "Fair product at a fair price",
    "It works fine not too impressed",
    "Okay product not bad not great",
    "Satisfactory purchase meets expectations",
    "It arrived on time and works okay",
    "Not the best but not the worst",
    "Does what it says nothing extra",
    "Reasonable product for the cost",
    "Acceptable quality for everyday use",
    "Standard product no complaints no praise",
]

texts  = positive_texts + negative_texts + neutral_texts
labels = [1]*len(positive_texts) + [0]*len(negative_texts) + [2]*len(neutral_texts)

df = pd.DataFrame({"text": texts, "label": labels})
print(f"\n Total samples   : {len(df)}")
print(f" Positive (1)    : {sum(df['label']==1)}")
print(f" Negative (0)    : {sum(df['label']==0)}")
print(f" Neutral  (2)    : {sum(df['label']==2)}")

# ── Split ──
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"\n Train: {len(X_train)} | Test: {len(X_test)}")

# ── TF-IDF ──
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)
print(f" Vocabulary size  : {len(vectorizer.vocabulary_)}")

# ── Train ──
model = LogisticRegression(max_iter=2000, C=2.0, solver="lbfgs", random_state=42)
model.fit(X_train_tfidf, y_train)
print("\n Model training complete!")

# ── Evaluate ──
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy : {accuracy*100:.2f}%")
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative","Positive","Neutral"]))

# ── Save ──
with open("model.pkl","wb") as f: pickle.dump(model, f)
with open("vectorizer.pkl","wb") as f: pickle.dump(vectorizer, f)
print(" model.pkl      -> saved")
print(" vectorizer.pkl -> saved")

# ── Quick Test ──
label_map = {1:"Positive", 0:"Negative", 2:"Neutral"}
test_sentences = [
    "This product is absolutely amazing and I love it",
    "Terrible quality, very disappointed with the purchase",
    "It is okay, nothing special, works fine",
]
print("\n Quick Tests:")
for s in test_sentences:
    vec  = vectorizer.transform([s])
    pred = model.predict(vec)[0]
    conf = max(model.predict_proba(vec)[0])*100
    print(f"  [{label_map[pred]} {conf:.0f}%] {s}")

print("\n Done! Run: python app.py")