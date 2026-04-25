import streamlit as st
import joblib
import numpy as np
from datetime import datetime
import os
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heartio – Heart Disease Detection",
    page_icon="❤️",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
USERNAME = "admin"
PASSWORD = "12345678"
PREDICTIONS_DIR = "predictions"

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model_2.pkl")

model = load_model()

# ── Session-state helpers ─────────────────────────────────────────────────────
def is_logged_in() -> bool:
    return st.session_state.get("user") is not None

def logout():
    st.session_state.pop("user", None)
    st.rerun()

# ── Persistence helpers ───────────────────────────────────────────────────────
def save_prediction(features: dict, result: str):
    user = st.session_state.get("user", "anonymous")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    prediction_id = f"{user}_{timestamp}"
    data = {
        "id": prediction_id,
        "user": user,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
        "features": features,
    }
    with open(os.path.join(PREDICTIONS_DIR, f"{prediction_id}.json"), "w") as f:
        json.dump(data, f, indent=4)


def get_predictions() -> list:
    user = st.session_state.get("user", "anonymous")
    records = []
    if not os.path.exists(PREDICTIONS_DIR):
        return records
    for fname in os.listdir(PREDICTIONS_DIR):
        if fname.startswith(f"{user}_") and fname.endswith(".json"):
            with open(os.path.join(PREDICTIONS_DIR, fname)) as f:
                records.append(json.load(f))
    records.sort(key=lambda x: x["date"], reverse=True)
    return records


def delete_prediction(prediction_id: str):
    path = os.path.join(PREDICTIONS_DIR, f"{prediction_id}.json")
    if os.path.exists(path):
        os.remove(path)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def render_login():
    st.title("❤️ Heartio – Heart Disease Detection")
    st.subheader("Login")

    st.info("💡 **Demo credentials** — Username: `admin` | Password: `12345678`")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username.strip() and password.strip():
            st.session_state["user"] = username.strip()
            st.rerun()
        else:
            st.error("Please enter both username and password.")


def render_dashboard():
    st.title("❤️ Heartio Dashboard")
    st.write(f"Welcome, **{st.session_state['user']}**!")
    st.markdown("---")
    st.info(
        "Use the **sidebar** to navigate between pages: "
        "Predict, History, Profile, Insights, Videos, and Lifestyle."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", "~85%")
    col2.metric("Features Used", "13")
    col3.metric("Dataset Records", "303")


def render_predict():
    st.title("🫀 Heart Disease Prediction")
    st.markdown("Fill in the patient's medical parameters below.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age      = st.number_input("Age",            min_value=1,  max_value=120, value=45)
            sex      = st.selectbox("Sex",               options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp       = st.selectbox("Chest Pain Type",   options=[0, 1, 2, 3],
                                    format_func=lambda x: ["Typical Angina","Atypical Angina","Non-anginal","Asymptomatic"][x])
            trestbps = st.number_input("Resting BP (mm Hg)",   min_value=50,  max_value=250, value=120)
            chol     = st.number_input("Cholesterol (mg/dl)",   min_value=50,  max_value=600, value=200)

        with col2:
            fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            restecg  = st.selectbox("Resting ECG",       options=[0, 1, 2],
                                    format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
            thalach  = st.number_input("Max Heart Rate Achieved", min_value=50,  max_value=250, value=150)
            exang    = st.selectbox("Exercise Induced Angina",    options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with col3:
            oldpeak  = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope    = st.selectbox("Slope of Peak ST Segment",   options=[0, 1, 2],
                                    format_func=lambda x: ["Upsloping","Flat","Downsloping"][x])
            ca       = st.selectbox("No. of Major Vessels (0–3)", options=[0, 1, 2, 3])
            thal     = st.selectbox("Thalassemia",                options=[0, 1, 2, 3],
                                    format_func=lambda x: ["Unknown","Normal","Fixed Defect","Reversible Defect"][x])

        submitted = st.form_submit_button("Predict")

    if submitted:
        features_vec = [float(v) for v in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        arr = np.array([features_vec])
        prediction = model.predict(arr)

        output = "Low Risk" if prediction[0] == 1 else "High Risk"

        try:
            probability = model.predict_proba(arr)[0][1]
            prob_text = f" (Confidence: {round(probability * 100, 2)}%)"
        except Exception:
            prob_text = ""

        if output == "High Risk":
            st.error(f"🚨 Heart Disease Prediction: **{output}**{prob_text}")
        else:
            st.success(f"✅ Heart Disease Prediction: **{output}**{prob_text}")

        features_dict = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }
        save_prediction(features_dict, output)
        st.caption("Prediction saved to history.")


def render_history():
    st.title("📋 Prediction History")
    records = get_predictions()

    if not records:
        st.info("No predictions found. Make a prediction first!")
        return

    for rec in records:
        with st.expander(f"🗓 {rec['date']} — {rec['result']}"):
            st.json(rec["features"])
            if st.button("🗑 Delete", key=f"del_{rec['id']}"):
                delete_prediction(rec["id"])
                st.success("Deleted.")
                st.rerun()


def render_profile():
    st.title("👤 User Profile")
    st.write(f"**Logged in as:** {st.session_state.get('user', 'N/A')}")
    st.markdown(
        "Profile management (name, email, health info) can be added here in a future update."
    )


def render_insights():
    st.title("📊 Health Insights")
    st.markdown(
        """
        This section provides visual representations of risk factors and health parameters.

        - **Age** is one of the strongest predictors of heart disease.
        - **High cholesterol** (>200 mg/dl) significantly increases risk.
        - **Exercise-induced angina** is a strong indicator of coronary artery disease.
        - **Maximum heart rate** declines with age; lower values correlate with higher risk.
        - **ST depression (Oldpeak)** > 2 is associated with higher risk.

        Connect a dataset to this page for interactive charts in a future version.
        """
    )


def render_videos():
    st.title("🎥 Educational Videos")
    st.markdown(
        """
        Learn more about heart health through curated resources:

        - [Understanding Heart Disease – Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118)
        - [Heart Disease Prevention – WHO](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))
        - [AHA – Lifestyle Changes for Heart Health](https://www.heart.org/en/healthy-living)

        *(Video embeds can be added here using `st.video()`.)*
        """
    )


def render_lifestyle():
    st.title("🌿 Lifestyle Guidance")
    st.markdown(
        """
        ### Tips for a Heart-Healthy Lifestyle

        **Diet**
        - Eat more fruits, vegetables, and whole grains.
        - Limit saturated fats, trans fats, and sodium.
        - Reduce sugar-sweetened beverages.

        **Exercise**
        - Aim for at least 150 minutes of moderate aerobic activity per week.
        - Include strength training twice a week.

        **Habits**
        - Quit smoking — it doubles the risk of heart disease.
        - Limit alcohol consumption.
        - Manage stress through mindfulness or yoga.

        **Regular Check-ups**
        - Monitor blood pressure and cholesterol regularly.
        - Keep diabetes under control if applicable.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not is_logged_in():
        render_login()
        return

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## ❤️ Heartio")
        st.markdown(f"👤 **{st.session_state['user']}**")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["Dashboard", "Predict", "History", "Profile", "Insights", "Videos", "Lifestyle"],
        )
        st.markdown("---")
        if st.button("🚪 Logout"):
            logout()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Predict":
        render_predict()
    elif page == "History":
        render_history()
    elif page == "Profile":
        render_profile()
    elif page == "Insights":
        render_insights()
    elif page == "Videos":
        render_videos()
    elif page == "Lifestyle":
        render_lifestyle()


st.markdown("---")
st.markdown("Developed by Pranav V P as an Internship Project")

if __name__ == "__main__":
    main()


