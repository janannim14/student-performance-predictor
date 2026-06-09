# ============================================================
# STUDENT PERFORMANCE PREDICTOR
# Day 2: Flask Web App Backend
# Author: Jananni M
# ============================================================

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load Model Artifacts ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "model", "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "model", "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

with open(os.path.join(BASE_DIR, "model", "features.pkl"), "rb") as f:
    features = pickle.load(f)

print("✅ All model artifacts loaded successfully!")

# ── Encoding Maps (must match training data) ──────────────────
ENCODING_MAPS = {
    "gender": {"male": 1, "female": 0},
    "race_ethnicity": {
        "group A": 0, "group B": 1, "group C": 2,
        "group D": 3, "group E": 4
    },
    "parental_education": {
        "some high school": 4,
        "high school": 2,
        "some college": 5,
        "associate's degree": 0,
        "bachelor's degree": 1,
        "master's degree": 3
    },
    "lunch": {"standard": 1, "free/reduced": 0},
    "test_prep_course": {"completed": 0, "none": 1}
}


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Encode categorical inputs
        encoded = []
        for feat in features:
            val = data.get(feat)
            if feat in ENCODING_MAPS:
                encoded.append(ENCODING_MAPS[feat].get(val, 0))
            else:
                encoded.append(float(val) if val is not None else 0)

        # Scale & Predict
        input_array = np.array([encoded])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        prediction = round(float(prediction), 2)

        # Grade logic
        if prediction >= 90:
            grade, emoji, advice = "A+", "🏆", "Outstanding! Keep it up!"
        elif prediction >= 80:
            grade, emoji, advice = "A", "🌟", "Excellent performance!"
        elif prediction >= 70:
            grade, emoji, advice = "B", "✅", "Good job! A little more effort for A grade."
        elif prediction >= 60:
            grade, emoji, advice = "C", "📘", "Average. Focus on weak subjects."
        elif prediction >= 50:
            grade, emoji, advice = "D", "⚠️", "Below average. Consider extra practice."
        else:
            grade, emoji, advice = "F", "❌", "Needs significant improvement. Seek help."

        return jsonify({
            "success": True,
            "predicted_score": prediction,
            "grade": grade,
            "emoji": emoji,
            "advice": advice
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    print("\n🚀 Starting Student Performance Predictor App...")
    print("   Open browser → http://127.0.0.1:5000\n")
    app.run(debug=True)