import streamlit as st
import pandas as pd
import numpy as np
import pickle
from fetch_profile import fetch_instagram_profile, extract_features_from_profile, save_profile_picture
from pathlib import Path
import base64
import matplotlib.pyplot as plt

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
    .prediction-box {{
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }}
    .genuine {{
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }}
    .fake {{
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# === Load background image ===
set_background("new.png")

# === Load models and scaler ===
xgb_model = pickle.load(open("model/xgb_model.pkl", "rb"))
rf_model = pickle.load(open("model/rf_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
feature_columns = pickle.load(open("model/feature_order.pkl", "rb"))

# === Create Tabs ===
tab1, tab2, tab3 = st.tabs(["🚀 Let's Start", "ℹ️ About", "🔍 Detector"])

# ====================================================
# TAB 1: Let's Start
# ====================================================
with tab1:
    st.title("🚀 Welcome to the Instagram Fake Profile Detector")
    st.markdown("""
    ### 👋 Hello!
    Welcome to our AI-powered tool that helps identify **fake or suspicious Instagram profiles**.

    Using advanced **Machine Learning models (XGBoost & Random Forest)**, it analyzes profile data — followers, biography, privacy — to predict authenticity.

    ---
    """)
    if st.button("👉 Get Started"):
        st.success("✅ Great! Move to the **Detector tab** to analyze a profile.")

# ====================================================
# TAB 2: About (Compact)
# ====================================================
with tab2:
    st.title("ℹ️ About Project")
    st.markdown("""
    **🎯 Mission:**  
    To make Instagram safer by detecting fake accounts automatically.

    **🧠 How It Works:**  
    Fetches public profile data → Extracts features → Predicts genuineness using **XGBoost** & **Random Forest**.

    **💻 Built With:**  
    Streamlit • Scikit-learn • XGBoost • Random Forest

    ---
    **Developed with ❤️ by Team InstaScan**
    """)

# ====================================================
# TAB 3: Instagram Fake Profile Detector
# ====================================================
with tab3:
    st.title("🔍 Instagram Fake Profile Detection")

    username = st.text_input("Enter Instagram Username:")
    analyze_button = st.button("🚀 Analyze Profile")

    if analyze_button and username:
        profile = fetch_instagram_profile(username)

        if profile is None:
            st.error("❌ Could not fetch profile data.")
        else:
            profile_pic = save_profile_picture(profile.get("profile_pic_url", ""), username)
            bio = profile.get("biography", "")

            col1, col2 = st.columns([1, 3])

            # === Profile Picture (Fixed Display) ===
            with col1:
                if profile_pic and Path(profile_pic).exists():
                    st.image(str(profile_pic), width=150, caption="📸 Profile Picture")
                else:
                    try:
                        img_url = profile.get("profile_pic_url", "")
                        if img_url:
                            st.image(img_url, width=150, caption="📸 Profile Picture (from URL)")
                        else:
                            st.warning("⚠️ No profile picture available.")
                    except Exception as e:
                        st.warning(f"⚠️ Unable to load profile picture: {e}")

            # === Profile Basic Info ===
            with col2:
                st.markdown(f"### 👤 {profile.get('full_name', '')}")
                st.markdown(f"📝 _{bio}_")

            # === Profile Details ===
            st.markdown("#### 📌 Profile Details")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Followers", f"{profile.get('follower_count', 0):,}")
                st.metric("Following", f"{profile.get('following_count', 0):,}")
            with c2:
                st.metric("Private?", "🔐 Yes" if profile.get("is_private", 0) else "🌍 No")
                st.metric("Username Length", len(profile.get("username", "")))
            with c3:
                st.metric("Biography Length", len(bio))
                st.metric("User ID", profile.get("id", "-"))

            # === Extract Features ===
            features = extract_features_from_profile(profile)
            features["follower_following_ratio"] = features.get("edge_followed_by", 0) / (features.get("edge_follow", 0) + 1)
            features["following_to_follower_ratio"] = features.get("edge_follow", 0) / (features.get("edge_followed_by", 0) + 1)

            # Fill missing
            for col in feature_columns:
                if col not in features:
                    features[col] = 0

            feature_df = pd.DataFrame([features])[feature_columns]
            scaled_features = scaler.transform(feature_df)

            try:
                # Predictions
                xgb_proba = xgb_model.predict_proba(scaled_features)[0]
                rf_proba = rf_model.predict_proba(scaled_features)[0]

                xgb_pred = int(np.argmax(xgb_proba))
                rf_pred = int(np.argmax(rf_proba))
                xgb_conf = np.max(xgb_proba)
                rf_conf = np.max(rf_proba)

                best_model = "XGBoost" if xgb_conf >= rf_conf else "Random Forest"
                best_pred = xgb_pred if best_model == "XGBoost" else rf_pred
                best_proba = xgb_proba if best_model == "XGBoost" else rf_proba

                # === Prediction Results ===
                st.subheader("🧠 Prediction Results")

                def display_prediction(name, pred, proba, is_best=False):
                    label = "✅ Genuine" if pred == 0 else "❌ Fake"
                    css_class = "genuine" if pred == 0 else "fake"
                    highlight = " (Best Model)" if is_best else ""
                    st.markdown(
                        f'<div class="prediction-box {css_class}">{name}{highlight}<br>'
                        f"{label}<br>Confidence → Genuine: {proba[0]*100:.1f}%, Fake: {proba[1]*100:.1f}%</div>",
                        unsafe_allow_html=True
                    )

                display_prediction("XGBoost", xgb_pred, xgb_proba, best_model == "XGBoost")
                display_prediction("Random Forest", rf_pred, rf_proba, best_model == "Random Forest")

                st.success(f"📌 Final Decision: **{'✅ Genuine' if best_pred == 0 else '❌ Fake'}** "
                           f"(by {best_model}, {max(best_proba)*100:.1f}% confidence)")

                # === Metrics Table ===
                with st.expander("📈 Evaluation Metrics for this Profile"):
                    results_df = pd.DataFrame({
                        "Model": ["XGBoost", "Random Forest", "Final Decision"],
                        "Predicted Label": [
                            "Genuine" if xgb_pred == 0 else "Fake",
                            "Genuine" if rf_pred == 0 else "Fake",
                            "Genuine" if best_pred == 0 else "Fake"
                        ],
                        "Confidence (Genuine %)": [
                            f"{xgb_proba[0]*100:.1f}%",
                            f"{rf_proba[0]*100:.1f}%",
                            f"{best_proba[0]*100:.1f}%"
                        ],
                        "Confidence (Fake %)": [
                            f"{xgb_proba[1]*100:.1f}%",
                            f"{rf_proba[1]*100:.1f}%",
                            f"{best_proba[1]*100:.1f}%"
                        ]
                    })
                    st.dataframe(results_df, use_container_width=True)

                # === Probability Heatmaps ===
                def plot_probability_heatmap(proba, model_name):
                    fig, ax = plt.subplots(figsize=(1.5, 0.5), dpi=220)
                    ax.imshow([proba], cmap="Blues", aspect="auto")
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["Genuine", "Fake"], fontsize=6)
                    ax.set_yticks([])
                    for i, v in enumerate(proba):
                        ax.text(i, 0, f"{v*100:.0f}%", ha="center", va="center",
                                color="black", fontsize=7, fontweight="bold")
                    ax.set_title(model_name, fontsize=8, pad=1)
                    fig.tight_layout(pad=0.2)
                    st.pyplot(fig, clear_figure=True, use_container_width=False)

                with st.expander("📊 Model Prediction Probability Heatmaps"):
                    colh1, colh2 = st.columns(2)
                    with colh1:
                        plot_probability_heatmap(xgb_proba, "XGBoost")
                    with colh2:
                        plot_probability_heatmap(rf_proba, "Random Forest")

                with st.expander("🔍 Debug Info"):
                    st.write("Input Features:")
                    st.write(feature_df)
                    st.write("Scaled Features:")
                    st.write(scaled_features)

            except Exception as e:
                st.error(f"Prediction Error: {e}")

# ====================================================
# Footer Section
# ====================================================
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: rgba(0, 0, 0, 0.75);
    color: white;
    text-align: center;
    padding: 8px 0;
    font-size: 14px;
    border-top: 1px solid #444;
    backdrop-filter: blur(8px);
}
.footer span {
    font-weight: 600;
    color: #4CAF50;
}
</style>

<div class="footer">
    © Copyright 2025<br>
    <span>InstaScan AI</span> — Fake Social Media Accounts Detection
</div>
""", unsafe_allow_html=True)
