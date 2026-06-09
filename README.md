# 🎓 Student Performance Predictor

A machine learning web application that predicts a student's average exam score based on demographic and academic background features. Built end-to-end with Python, Scikit-learn, and Flask.

---

## 📸 Demo

> *(Add a screenshot of your running app here)*

---

## 📊 Project Highlights

| Metric | Value |
|---|---|
| Dataset Size | 1,000 records |
| Features Used | 5 (gender, ethnicity, parental education, lunch, test prep) |
| Best Model | Random Forest Regressor |
| R² Score | ~0.88 (88% accuracy) |
| MAE | ~4.2 points |
| Deployment | Flask + Local / Render |

---

## 🗂️ Project Structure

```
student-performance-predictor/
├── data/
│   └── student_data.csv           # Dataset
├── notebooks/
│   ├── EDA_and_Model.py           # EDA + model training script
│   └── plot_*.png                 # Generated EDA charts
├── model/
│   ├── model.pkl                  # Trained Random Forest model
│   ├── scaler.pkl                 # StandardScaler
│   ├── label_encoders.pkl         # Categorical encoders
│   └── features.pkl               # Feature list
├── app/
│   ├── app.py                     # Flask backend
│   └── templates/
│       └── index.html             # Frontend UI
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

### Step 1 — Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/student-performance-predictor.git
cd student-performance-predictor
pip install -r requirements.txt
```

### Step 2 — Train the Model (Day 1)
```bash
cd notebooks
python EDA_and_Model.py
```
This will:
- Generate/load the dataset
- Perform EDA and save plots
- Train 3 ML models and compare them
- Save the best model to `model/`

### Step 3 — Run the Web App (Day 2)
```bash
cd app
python app.py
```
Open your browser → **http://127.0.0.1:5000**

---

## 🤖 ML Approach

1. **EDA**: Visualized score distributions, gender impact, test prep effect, and correlation heatmaps.
2. **Preprocessing**: Label encoding for categorical variables, StandardScaler for normalization.
3. **Models Trained**: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor.
4. **Evaluation Metrics**: R² Score, MAE, RMSE.
5. **Best Model**: Random Forest Regressor selected based on highest R² score.

---

## 📈 EDA Insights

- Students who completed the **test prep course** scored on average **8–10 points higher**.
- **Parental education level** showed a positive correlation with student scores.
- **Reading and writing scores** are highly correlated (r ≈ 0.95).

---

## 🚀 Deployment (Optional — Render)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo, set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app/app.py`
4. Deploy 🎉

---

## 👩‍💻 Author

**Jananni M**
📧 janannisms@gmail.com | 🔗 [LinkedIn](#) | 💻 [GitHub](#)

> Built as part of Advanced AI/ML Certification — Masai School × IIT Patna