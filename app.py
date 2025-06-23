import streamlit as st
import pandas as pd
import numpy as np
import pickle
from fetch_profile import fetch_instagram_profile, extract_features_from_profile, save_profile_picture
from pathlib import Path
import base64
from sklearn.metrics import classification_report

# === Set Page Config FIRST ===
st.set_page_config(page_title="Instagram Fake Profile Detector", layout="wide")

# === Optional: Set Background Image ===
def set_background(image_file_path: str):
    with open(image_file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load background image
set_background("rb.jpg")

# === Load models and scaler ===
xgb_model = pickle.load(open("model/xgb_model.pkl", "rb"))
rf_model = pickle.load(open("model/rf_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
feature_columns = pickle.load(open("model/feature_order.pkl", "rb"))

# === Streamlit UI ===
st.title("🔍 Instagram Fake Profile Detection")

username = st.text_input("Enter Instagram Username:")
analyze_button = st.button("Analyze Profile")

if analyze_button and username:
    profile = fetch_instagram_profile(username)

    if profile is None:
        st.error("❌ Could not fetch profile data.")
    else:
        profile_pic = save_profile_picture(profile.get("profile_pic_url", ""), username)
        bio = profile.get("biography", "")

        if profile_pic:
            st.image(profile_pic, width=100, caption="📸 Profile Picture")

        st.markdown(f"### 👤 {profile.get('full_name', '')}")
        st.markdown(f"📝 Bio: _{bio}_")

        # === Basic Profile Information ===
        st.markdown("#### 📌 Profile Details:")
        st.markdown(f"- 👤 **Username:** `{profile.get('username', '')}`")
        st.markdown(f"- 🆔 **User ID:** `{profile.get('id', '-')}`")
        st.markdown(f"- 🔐 **Private Account:** {'Yes' if profile.get('is_private', 0) else 'No'}")
        st.markdown(f"- 🙋‍♂️ **Followers:** `{profile.get('follower_count', 0):,}`")
        st.markdown(f"- 🤝 **Following:** `{profile.get('following_count', 0):,}`")
        st.markdown(f"- 🧠 **Username Length:** `{len(profile.get('username', ''))}`")
        st.markdown(f"- 📝 **Biography Length:** `{len(bio)}`")

        features = extract_features_from_profile(profile)

        # === Derived features ===
        features["follower_following_ratio"] = features.get("edge_followed_by", 0) / (features.get("edge_follow", 0) + 1)
        features["following_to_follower_ratio"] = features.get("edge_follow", 0) / (features.get("edge_followed_by", 0) + 1)

        # === Fill missing feature columns ===
        for col in feature_columns:
            if col not in features:
                features[col] = 0

        feature_df = pd.DataFrame([features])[feature_columns]
        scaled_features = scaler.transform(feature_df)

        try:
            # === Model Predictions ===
            xgb_proba = xgb_model.predict_proba(scaled_features)[0]
            rf_proba = rf_model.predict_proba(scaled_features)[0]

            xgb_pred = int(np.argmax(xgb_proba))
            rf_pred = int(np.argmax(rf_proba))

            xgb_conf = np.max(xgb_proba)
            rf_conf = np.max(rf_proba)

            best_model = "XGBoost" if xgb_conf >= rf_conf else "Random Forest"
            best_pred = xgb_pred if best_model == "XGBoost" else rf_pred
            best_proba = xgb_proba if best_model == "XGBoost" else rf_proba

            def display_prediction(name, pred, proba, is_best=False):
                label = "✅ Genuine" if pred == 0 else "❌ Fake"
                style = "**" if is_best else ""
                st.markdown(f"{style}{name} Prediction:{style} {label}")
                st.markdown(f"• Confidence - Genuine: `{proba[0]*100:.2f}%`, Fake: `{proba[1]*100:.2f}%`")

            st.subheader("🧠 Prediction Results")
            display_prediction("XGBoost", xgb_pred, xgb_proba, best_model == "XGBoost")
            display_prediction("Random Forest", rf_pred, rf_proba, best_model == "Random Forest")

            st.success(f"📌 Based on confidence, **{best_model}** is selected.")
            st.info(f"🔎 Final Decision: **{'✅ Genuine' if best_pred == 0 else '❌ Fake'}** (Confidence: `{max(best_proba)*100:.2f}%`)")

            with st.expander("🔍 Debug Info"):
                st.write("Input Features:")
                st.write(feature_df)
                st.write("Scaled Features:")
                st.write(scaled_features)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# === Evaluation Metrics on Test Set (Static/Precomputed) ===
with st.expander("📈 Evaluation Metrics on Test Set"):
    try:
        with open("model/metrics.pkl", "rb") as f:
            metrics = pickle.load(f)

        st.markdown("### ✅ Model Evaluation on Test Set")
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        st.metric("Precision", f"{metrics['precision']*100:.2f}%")
        st.metric("Recall", f"{metrics['recall']*100:.2f}%")
        st.metric("F1 Score", f"{metrics['f1']*100:.2f}%")

        # Optional: Show confusion matrix
        st.markdown("#### Confusion Matrix")
        cm = metrics.get("conf_matrix", [[0, 0], [0, 0]])
        cm_df = pd.DataFrame(cm, columns=["Predicted Genuine", "Predicted Fake"],
                                  index=["Actual Genuine", "Actual Fake"])
        st.dataframe(cm_df)

    except Exception as e:
        st.error(f"❌ Could not load evaluation metrics: {e}")


# === Confusion Matrices (Images) ===
with st.expander("📊 View Model Confusion Matrices"):
    st.image("model/xgb_confusion_matrix.png", caption="XGBoost Confusion Matrix")
    st.image("model/rf_confusion_matrix.png", caption="Random Forest Confusion Matrix")
