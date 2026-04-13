# ============================================================
#  Sentiment Analysis - Flask Web Application
#  Technique: NLP + Logistic Regression
#  Deployment: AWS EC2
# ============================================================

from flask import Flask, render_template, request
import pickle
import numpy as np
import re

app = Flask(__name__)

# ── Load trained model and vectorizer ──
model      = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ── Label mapping ──
label_map = {
    0: ("Negative", "😞", "#e74c3c"),
    1: ("Positive", "😊", "#2ecc71"),
    2: ("Neutral",  "😐", "#f39c12"),
}

# ── Known vocabulary from training ──
KNOWN_VOCAB = set(vectorizer.vocabulary_.keys())

def validate_input(text):
    """
    Returns (is_valid: bool, error_message: str)
    Checks for: empty, too short, numbers only, gibberish,
    special chars only, and no recognizable English words.
    """

    # 1. Empty check
    if not text or not text.strip():
        return False, "⚠ Please enter some text before analyzing."

    text = text.strip()

    # 2. Too short (less than 3 characters)
    if len(text) < 3:
        return False, "⚠ Input too short. Please enter a meaningful sentence or phrase."

    # 3. Only numbers / digits
    if re.fullmatch(r"[\d\s\.\,]+", text):
        return False, "⚠ Invalid input. Please enter text, not just numbers."

    # 4. Only special characters / symbols
    if re.fullmatch(r"[^a-zA-Z0-9\s]+", text):
        return False, "⚠ Invalid input. Please enter proper English text."

    # 5. Extract only alphabetic words
    words = re.findall(r"[a-zA-Z]+", text.lower())

    # 6. No alphabetic words at all
    if len(words) == 0:
        return False, "⚠ Invalid input. Please enter text with actual words."

    # 7. Single character words / gibberish letters only
    real_words = [w for w in words if len(w) >= 2]
    if len(real_words) == 0:
        return False, "⚠ Invalid input. Please type a proper sentence or phrase."

    # 8. Check if at least 1 word exists in our trained vocabulary
    #    (removes completely gibberish inputs like "asdfgh xyzxyz")
    matched = [w for w in real_words if w in KNOWN_VOCAB]
    if len(matched) == 0:
        return False, (
            "⚠ I could not understand your input. "
            "Please enter a proper English sentence related to a product, "
            "review, or opinion (e.g. 'The product is good' or 'Very bad quality')."
        )

    # 9. Mostly gibberish: less than 20% of words are recognized
    match_ratio = len(matched) / len(real_words)
    if match_ratio < 0.2 and len(real_words) > 3:
        return False, (
            "⚠ Your input contains too many unrecognized words. "
            "Please enter a clear English sentence."
        )

    # 10. Repeated single character spam (e.g. "aaaaaaa" or "hhhhhh")
    for w in real_words:
        if len(w) > 4 and len(set(w)) == 1:
            return False, "⚠ Invalid input. Please type a proper meaningful sentence."

    return True, ""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form.get("review", "").strip()

    # ── Validate input ──
    is_valid, error_msg = validate_input(user_text)
    if not is_valid:
        return render_template("index.html", error=error_msg, user_text=user_text)

    # ── Transform using TF-IDF ──
    text_vector   = vectorizer.transform([user_text])
    prediction    = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]

    sentiment, emoji, color = label_map[prediction]

    classes   = model.classes_
    prob_dict = {int(c): round(float(p) * 100, 1) for c, p in zip(classes, probabilities)}
    neg_conf  = prob_dict.get(0, 0)
    pos_conf  = prob_dict.get(1, 0)
    neu_conf  = prob_dict.get(2, 0)
    confidence = round(max(probabilities) * 100, 1)

    return render_template(
        "index.html",
        prediction_text = sentiment,
        emoji           = emoji,
        color           = color,
        confidence      = confidence,
        user_text       = user_text,
        neg_conf        = neg_conf,
        pos_conf        = pos_conf,
        neu_conf        = neu_conf,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)