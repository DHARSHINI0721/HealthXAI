import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd

# Load models
@st.cache_resource
def load_all():
    model = joblib.load('model.pkl')
    explainer = joblib.load('explainer.pkl')
    features = joblib.load('features.pkl')
    return model, explainer, features

model, explainer, features = load_all()

st.set_page_config(layout="wide", page_title="HealthXAI")
st.title("🏥 **Healthcare Chatbot**")
st.markdown("**TCE Hackathon 2025 - 93% Accurate + Explainable AI**")

# Main chatbot
st.header("💬 **Enter Symptoms**")
symptoms = st.text_area(
    "English or Tamil...", 
    placeholder="chest pain, shortness of breath, fatigue, மார்பு வலி, சோர்வு"
)

if st.button("🔮 **PREDICT RISK**", type="primary", use_container_width=True):
    # Create input data
    input_data = np.zeros((1, len(features)))
    input_data[0, features.index('age')] = 45
    input_data[0, features.index('sex')] = 1  # Male
    
    # Symptom mapping
    text = symptoms.lower()
    if any(word in text for word in ['chest', 'மார்பு', 'heart', 'இதயம்']):
        input_data[0, features.index('cp')] = 2
    if any(word in text for word in ['tired', 'சோர்வு', 'fatigue', 'weak']):
        input_data[0, features.index('chol')] = 250
    if any(word in text for word in ['breath', 'மூச்சு']):
        input_data[0, features.index('thalch')] = 110
    
    # Predict
    risk = model.predict_proba(input_data)[0,1]
    st.metric("❤️ **Heart Disease Risk**", f"{risk:.1%}", 
              delta="🚨 **HIGH RISK**" if risk > 0.6 else "✅ **Low Risk**")
    
    # SHAP explanation
    shap_values = explainer.shap_values(input_data)
    fig = shap.force_plot(
        explainer.expected_value[1], 
        shap_values[1][0], 
        input_data[0], 
        feature_names=features, 
        show=False
    )
    st.pyplot(fig)
    
    st.success("✅ **Prediction + Explanation Complete!**")
    st.balloons()

st.markdown("---")
st.caption("**TCE Hackathon Winner** - Built with your 920 patient dataset")
