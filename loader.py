# --- 📦 Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import xgboost as xgb
import instaloader
from PIL import Image
import requests
from io import BytesIO
import os

# --- 🎨 Page Configuration ---
st.set_page_config(page_title="Instagram Fake Profile Detector", layout="wide")
st.title("🔍 Instagram Fake Profile Detection using XGBoost")

# --- 📥 Load Model & Feature Setup ---
xgb_model = pickle.load(open("model/xgb_model.pkl", "rb"))

# 🛠️ Manually define the feature columns expected from Instagram extraction:
feature_columns = [
    'edge_followed_by',
    'edge_follow',
    'statuses_count',
    'default_profile',
    'is_verified',
    'biography_length'
]

# --- 🎛️ Input Section ---
input_col, button_col = st.columns([2, 1])
with input_col:
    input_username = st.text_input("Instagram Username", "nandanpatil122", key="username_input")
with button_col:
    st.write("")
    analyze_button = st.button("Analyze Profile", key="analyze_button")

# --- 🚀 Helper Functions ---
def fetch_instagram_profile(username):
    try:
        loader = instaloader.Instaloader()
        profile = instaloader.Profile.from_username(loader.context, username)
        return {
            "username": profile.username,
            "full_name": profile.full_name,
            "biography": profile.biography,
            "followers": profile.followers,
            "following": profile.followees,
            "mediacount": profile.mediacount,
            "profile_pic_url": profile.profile_pic_url,
            "is_private": profile.is_private,
            "is_verified": profile.is_verified,
        }
    except Exception as e:
        st.error(f"Error fetching profile: {e}")
        return None

def save_profile_picture(url, username):
    try:
        if not os.path.exists("profile_pics"):
            os.makedirs("profile_pics")
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        file_path = f"profile_pics/{username}.jpg"
        img.save(file_path)
        return file_path
    except:
        return None

def extract_features_from_profile(profile_json):
    return {
        'edge_followed_by': profile_json.get("followers", 0),
        'edge_follow': profile_json.get("following", 0),
        'statuses_count': profile_json.get("mediacount", 0),
        'default_profile': 1 if profile_json.get("is_private", False) else 0,
        'is_verified': 1 if profile_json.get("is_verified", False) else 0,
        'biography_length': len(profile_json.get("biography", "")),
    }

# --- 🚀 Main Analysis ---
if analyze_button:
    user_profile = fetch_instagram_profile(input_username)

    if user_profile is None:
        st.error("❌ Could not fetch profile data.")
    else:
        # --- Save Profile Picture ---
        profile_pic_path = save_profile_picture(user_profile.get("profile_pic_url", ""), input_username)
        bio_text = user_profile.get("biography", "")

        if profile_pic_path:
            pic_col, info_col = st.columns([1, 3])
            with pic_col:
                st.image(profile_pic_path, width=100, caption="📸 Profile Picture")
            with info_col:
                full_name = user_profile.get("full_name", "N/A")
                st.markdown(f"### 👤 {full_name}")
                st.markdown(f"📝 **Bio:** _{bio_text or 'No bio available'}_")
        else:
            st.warning("⚠️ Profile picture could not be loaded.")

        # --- 📈 Feature Extraction ---
        extracted_features = extract_features_from_profile(user_profile)
        feature_df = pd.DataFrame([extracted_features])[feature_columns]

        # --- 🧠 Prediction ---
        prediction_label = xgb_model.predict(feature_df)[0]
        class_probs = xgb_model.predict_proba(feature_df)[0]
        confidence_percent = class_probs[int(prediction_label)] * 100

        prediction_text = "✅ Genuine" if prediction_label == 0 else "❌ Fake"
        st.subheader(f"Prediction Result: {prediction_text}")
        st.write(f"- 🔥 Confidence: **{confidence_percent:.2f}%**")

        # --- 📊 Show Extracted Features ---
        st.markdown("---")
        st.markdown("### 📊 Extracted Profile Features")
        st.dataframe(feature_df)

        # --- 🤔 Reasoning Behind Prediction ---
        st.markdown("### 🤔 Reasoning Behind Prediction")
        reasoning = ""
        if feature_df['edge_followed_by'].values[0] < 100 or feature_df['statuses_count'].values[0] < 5:
            reasoning += "- Low followers or very few posts.\n"
        if feature_df['default_profile'].values[0] == 1:
            reasoning += "- Default/private profile settings.\n"
        if feature_df['edge_follow'].values[0] > 1000:
            reasoning += "- Following unusually high number of users.\n"

        st.code(reasoning if reasoning else "Profile characteristics align with a genuine user.")
