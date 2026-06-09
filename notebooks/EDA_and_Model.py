# ============================================================
# STUDENT PERFORMANCE PREDICTOR
# Day 1: EDA + Model Training
# Author: Jananni M
# ============================================================

# ── 1. INSTALL & IMPORT LIBRARIES ───────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings("ignore")

print("✅ All libraries imported successfully!\n")


# ── 2. LOAD DATASET ─────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
# Save as: data/student_data.csv
# OR use the synthetic dataset generator below if you don't have Kaggle access

def generate_synthetic_data(n=1000):
    """Generate realistic synthetic student data for demo purposes."""
    np.random.seed(42)
    data = {
        "gender":              np.random.choice(["male", "female"], n),
        "race_ethnicity":      np.random.choice(["group A","group B","group C","group D","group E"], n),
        "parental_education":  np.random.choice(["some high school","high school","some college",
                                                  "associate's degree","bachelor's degree","master's degree"], n),
        "lunch":               np.random.choice(["standard", "free/reduced"], n),
        "test_prep_course":    np.random.choice(["none", "completed"], n),
        "math_score":          np.clip(np.random.normal(66, 15, n).astype(int), 0, 100),
        "reading_score":       np.clip(np.random.normal(69, 14, n).astype(int), 0, 100),
        "writing_score":       np.clip(np.random.normal(68, 15, n).astype(int), 0, 100),
    }
    df = pd.DataFrame(data)
    # Add target: average score
    df["average_score"] = ((df["math_score"] + df["reading_score"] + df["writing_score"]) / 3).round(2)
    return df

# Try loading real data, fall back to synthetic
import os
DATA_PATH = "data/student_data.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Real dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    df = generate_synthetic_data(1000)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Synthetic dataset generated & saved: {df.shape[0]} rows")
    print("   (Download real dataset from Kaggle for production use)\n")


# ── 3. EXPLORATORY DATA ANALYSIS (EDA) ──────────────────────
print("=" * 55)
print("📊  EXPLORATORY DATA ANALYSIS")
print("=" * 55)

print("\n🔹 First 5 rows:")
print(df.head())

print(f"\n🔹 Shape: {df.shape}")
print(f"\n🔹 Missing Values:\n{df.isnull().sum()}")
print(f"\n🔹 Data Types:\n{df.dtypes}")
print(f"\n🔹 Basic Statistics:\n{df.describe().round(2)}")

# ── Plot 1: Score Distributions ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Score Distributions", fontsize=14, fontweight="bold")

score_cols = [c for c in ["math_score","reading_score","writing_score","average_score"] if c in df.columns]
plot_cols  = score_cols[:3]

colors = ["#4C72B0", "#55A868", "#C44E52"]
for ax, col, color in zip(axes, plot_cols, colors):
    ax.hist(df[col], bins=20, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(df[col].mean(), color="black", linestyle="--", label=f"Mean: {df[col].mean():.1f}")
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
plt.savefig("notebooks/plot_score_distributions.png", dpi=120, bbox_inches="tight")
plt.show()
print("📈 Plot saved: plot_score_distributions.png")

# ── Plot 2: Gender vs Average Score ─────────────────────────
if "gender" in df.columns and "average_score" in df.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="gender", y="average_score", palette=["#4C72B0","#C44E52"])
    plt.title("Average Score by Gender", fontweight="bold")
    plt.xlabel("Gender")
    plt.ylabel("Average Score")
    plt.tight_layout()
    plt.savefig("notebooks/plot_gender_vs_score.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("📈 Plot saved: plot_gender_vs_score.png")

# ── Plot 3: Test Prep vs Score ───────────────────────────────
if "test_prep_course" in df.columns and "average_score" in df.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="test_prep_course", y="average_score", palette="Set2")
    plt.title("Impact of Test Prep Course on Average Score", fontweight="bold")
    plt.xlabel("Test Prep Course")
    plt.ylabel("Average Score")
    plt.tight_layout()
    plt.savefig("notebooks/plot_testprep_vs_score.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("📈 Plot saved: plot_testprep_vs_score.png")

# ── Plot 4: Correlation Heatmap ──────────────────────────────
numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) >= 2:
    plt.figure(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, square=True)
    plt.title("Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig("notebooks/plot_correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("📈 Plot saved: plot_correlation_heatmap.png")


# ── 4. FEATURE ENGINEERING & PREPROCESSING ──────────────────
print("\n" + "=" * 55)
print("⚙️   FEATURE ENGINEERING")
print("=" * 55)

df_model = df.copy()

# Encode categorical columns
le = LabelEncoder()
categorical_cols = df_model.select_dtypes(include=["object"]).columns.tolist()
print(f"\n🔹 Encoding categorical columns: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le_col = LabelEncoder()
    df_model[col] = le_col.fit_transform(df_model[col])
    label_encoders[col] = le_col
    print(f"   ✅ {col} → {list(le_col.classes_)}")

# Define features & target
# Target = average_score (or math_score if average not available)
TARGET = "average_score" if "average_score" in df_model.columns else "math_score"
DROP_COLS = ["math_score","reading_score","writing_score","average_score"]
FEATURES = [c for c in df_model.columns if c not in DROP_COLS]

print(f"\n🔹 Target Variable : {TARGET}")
print(f"🔹 Feature Columns : {FEATURES}")

X = df_model[FEATURES]
y = df_model[TARGET]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n🔹 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("✅ Feature scaling complete.")


# ── 5. MODEL TRAINING & COMPARISON ──────────────────────────
print("\n" + "=" * 55)
print("🤖  MODEL TRAINING & EVALUATION")
print("=" * 55)

models = {
    "Linear Regression":        LinearRegression(),
    "Random Forest Regressor":  RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":        GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {"R² Score": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4), "model": model}
    print(f"\n  📌 {name}")
    print(f"     R² Score : {r2:.4f}  ({r2*100:.1f}% accuracy)")
    print(f"     MAE      : {mae:.4f}")
    print(f"     RMSE     : {rmse:.4f}")

# ── Pick Best Model ──────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["R² Score"])
best_model = results[best_name]["model"]
print(f"\n🏆 Best Model: {best_name}  (R² = {results[best_name]['R² Score']})")

# ── Plot 5: Model Comparison Bar Chart ──────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
model_names = list(results.keys())
r2_scores   = [results[m]["R² Score"] for m in model_names]
bars = ax.barh(model_names, r2_scores, color=["#4C72B0","#55A868","#C44E52"], edgecolor="white")
ax.set_xlabel("R² Score")
ax.set_title("Model Comparison — R² Score", fontweight="bold")
for bar, score in zip(bars, r2_scores):
    ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2,
            f"{score:.4f}", va="center", ha="right", color="white", fontweight="bold")
ax.set_xlim(0, 1.05)
plt.tight_layout()
plt.savefig("notebooks/plot_model_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
print("📈 Plot saved: plot_model_comparison.png")

# ── Plot 6: Feature Importance (Random Forest) ───────────────
if "Random Forest Regressor" in results:
    rf_model = results["Random Forest Regressor"]["model"]
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True)

    plt.figure(figsize=(8, 4))
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#4C72B0", edgecolor="white")
    plt.title("Feature Importance — Random Forest", fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("notebooks/plot_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("📈 Plot saved: plot_feature_importance.png")


# ── 6. SAVE MODEL & SCALER ───────────────────────────────────
print("\n" + "=" * 55)
print("💾  SAVING MODEL & ARTIFACTS")
print("=" * 55)

with open("model/model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"✅ Best model saved → model/model.pkl  ({best_name})")

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved      → model/scaler.pkl")

with open("model/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
print("✅ Encoders saved    → model/label_encoders.pkl")

with open("model/features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)
print("✅ Feature list saved → model/features.pkl")

print("\n" + "=" * 55)
print("🎉  DAY 1 COMPLETE!")
print("=" * 55)
print(f"""
Summary:
  • Dataset        : {df.shape[0]} records
  • Features used  : {len(FEATURES)}
  • Best Model     : {best_name}
  • R² Score       : {results[best_name]['R² Score']} ({results[best_name]['R² Score']*100:.1f}% accuracy)
  • MAE            : {results[best_name]['MAE']}

Next → Run app/app.py for the Flask web app (Day 2)
""")