# Heartio – Heart Disease Detection System

![Heart Disease Detection](https://img.shields.io/badge/Health-Heart%20Disease%20Detection-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

Heartio is a **machine learning powered web application** that predicts the risk of heart disease using patient medical parameters. It uses a **Logistic Regression** model trained on the UCI Heart Disease dataset and is deployed as an interactive **Streamlit** app.

---

## Features

- **User Authentication** – Session-based login to protect user data
- **Heart Disease Prediction** – 13-feature ML model with instant risk output and confidence score
- **Prediction History** – View and delete past predictions stored as JSON files
- **User Profile** – Displays current logged-in user info
- **Health Insights** – Key risk factor explanations
- **Educational Resources** – Curated links to heart health articles
- **Lifestyle Guidance** – Diet, exercise, and habit tips for heart health

---

## Installation

### Prerequisites

- Python **3.8 or higher**
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/VPPranav/Heartio_Heart_Disease_Detection_WebApp.git
cd Heartio_Heart_Disease_Detection_WebApp

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

Open your browser and visit:

```
http://localhost:8501
```

### Demo Credentials

| Field    | Value      |
|----------|------------|
| Username | `admin`    |
| Password | `12345678` |

---

## Navigation

After login, use the **left sidebar** to move between pages:

| Page       | Description                                      |
|------------|--------------------------------------------------|
| Dashboard  | Welcome screen with model stats                  |
| Predict    | Enter patient parameters and get a risk result   |
| History    | View and delete past predictions                 |
| Profile    | Logged-in user info                              |
| Insights   | Key heart disease risk factor summaries          |
| Videos     | Curated educational resource links               |
| Lifestyle  | Heart-healthy diet, exercise, and habit tips     |

---

## Project Structure

```
├── predictions/                  ← Auto-created; stores per-user prediction JSON files
├── app.py                        ← Streamlit application (main entry point)
├── heart_disease_model_2.pkl     ← Trained Logistic Regression model
├── requirements.txt              ← Python dependencies
└── notebooks/
    └── heart_disease_detection_training_2.ipynb  ← Full ML training pipeline
```

---

## Machine Learning Pipeline

The model was built in `heart_disease_detection_training_2.ipynb` and follows this workflow:

**1. Problem Statement**
Predict whether a patient has heart disease based on 13 medical attributes. Early detection helps doctors take preventive action.

**2. Dataset**
- 303 patient records, 14 columns (13 features + 1 target)
- Source: UCI Heart Disease / Kaggle (`heart_disease_data.csv`)
- No missing values, minimal duplicates

**3. Exploratory Data Analysis**
- Target distribution — dataset is relatively balanced
- Correlation heatmap — identifies features most associated with heart disease
- Univariate analysis — age, cholesterol, and resting BP distributions
- Bivariate analysis — age vs. cholesterol, age vs. max heart rate, chest pain type vs. target
- Multivariate pairplot — relationships across key features by target class

Key findings from EDA:
- Chest pain type is a strong predictor of heart disease
- Maximum heart rate decreases with age and correlates with disease risk
- Certain chest pain categories appear far more frequently in positive cases

**4. Preprocessing**

```
Features (X): age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal

Target (Y): target  →  0 = No Heart Disease  |  1 = Heart Disease Present
```

- 80/20 train-test split with `stratify=Y` and `random_state=2`

**5. Model**
- Algorithm: **Logistic Regression** (`max_iter=1000`)
- Library: scikit-learn

**6. Evaluation**
- Accuracy Score
- Classification Report (Precision, Recall, F1-Score per class)
- Confusion Matrix

**7. Saved Model**
```
heart_disease_model_2.pkl    ← serialized with pickle, loaded by app.py via joblib
```

---

## Prediction Logic

The app loads `heart_disease_model_2.pkl` and applies this label mapping:

```python
if prediction[0] == 1:
    output = "Low Risk"
else:
    output = "High Risk"
```

> Note: The label swap (1 → Low Risk, 0 → High Risk) accounts for how the original dataset's target encoding was interpreted during training.

A confidence score is also shown when `predict_proba` is available.

---

## Dataset Features

| Feature    | Description                                  |
|------------|----------------------------------------------|
| `age`      | Age of the patient                           |
| `sex`      | Sex (0 = Female, 1 = Male)                   |
| `cp`       | Chest pain type (0–3)                        |
| `trestbps` | Resting blood pressure (mm Hg)               |
| `chol`     | Serum cholesterol (mg/dl)                    |
| `fbs`      | Fasting blood sugar > 120 mg/dl (0/1)        |
| `restecg`  | Resting ECG results (0–2)                    |
| `thalach`  | Maximum heart rate achieved                  |
| `exang`    | Exercise induced angina (0/1)                |
| `oldpeak`  | ST depression induced by exercise            |
| `slope`    | Slope of peak exercise ST segment (0–2)      |
| `ca`       | Number of major vessels colored by fluoroscopy (0–3) |
| `thal`     | Thalassemia (0–3)                            |
| `target`   | Heart disease presence (0 = No, 1 = Yes)     |

---

## Dependencies

```
streamlit>=1.35.0
numpy==1.25.2
scikit-learn==1.3.0
joblib==1.3.2
pandas==2.1.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Future Enhancements

- User registration system
- Interactive charts in the Insights page using the real dataset
- PDF report generation for predictions
- Email alerts for high-risk results
- Mobile application version
- Integration with wearable health devices

---

## Dataset Source

**Heart Disease Dataset** — Kaggle
Author: Redwankarimsony
Link: [kaggle.com/datasets/redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

---

## Acknowledgments

- UCI Machine Learning Repository
- scikit-learn
- Streamlit
- Kaggle

---

## Contact

```
pranavvp1507@gmail.com
```