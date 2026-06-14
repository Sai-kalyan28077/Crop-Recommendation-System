import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem; }
    .main-header h1 { color: #2E7D32; font-size: 2.5rem; }
    .prediction-box { padding: 1.5rem; border-radius: 10px; text-align: center; }
    .crop-name { font-size: 2rem; font-weight: bold; color: #1B5E20; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; text-align: center; }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #2E7D32; }
    .metric-label { font-size: 0.9rem; color: #666; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🌾 Crop Recommendation System</h1></div>',
            unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; font-size:1.1rem; color:#555;">
AI-powered system that recommends the most suitable crop based on your soil and environmental conditions.
</p>
""", unsafe_allow_html=True)

base_dir = os.path.dirname(__file__)
try:
    model = joblib.load(os.path.join(base_dir, 'crop_recommendation_model.pkl'))
    scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
except:
    st.error("Model files not found. Run model_training.py first.")
    st.stop()

st.sidebar.header("📊 Input Parameters")
st.sidebar.markdown("Adjust the sliders to match your soil conditions.")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 145, 50)
K = st.sidebar.slider("Potassium (K)", 0, 210, 50)
temperature = st.sidebar.slider("Temperature (°C)", 8.0, 45.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 14.0, 100.0, 60.0)
ph = st.sidebar.slider("pH Value", 3.5, 10.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 150.0)

input_data = pd.DataFrame({
    'N': [N], 'P': [P], 'K': [K],
    'temperature': [temperature], 'humidity': [humidity],
    'ph': [ph], 'rainfall': [rainfall]
})

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Input Soil & Weather Conditions")
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.subheader("ℹ️ Instructions")
    st.info("""
    1. Adjust the values on the left sidebar
    2. Click **Predict Crop** below
    3. Get an instant recommendation!
    """)

if st.button("🌱 Predict Recommended Crop", type="primary", use_container_width=True):
    inp = scaler.transform(input_data.values)
    pred = model.predict(inp)
    probs = model.predict_proba(inp)[0]

    crop = label_encoder.inverse_transform(pred)[0]
    conf = np.max(probs) * 100

    st.markdown("---")
    x, y, z = st.columns([1, 2, 1])
    with y:
        st.markdown(f"""
        <div class="prediction-box" style="background: linear-gradient(135deg, #E8F5E9, #C8E6C9);">
            <p style="font-size:1.2rem; color:#555;">Recommended Crop</p>
            <p class="crop-name">🌿 {crop.upper()}</p>
            <p style="font-size:1.1rem; color:#388E3C;">
                Confidence: {conf:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Prediction Probabilities for All Crops")
    prob_df = pd.DataFrame({
        'Crop': label_encoder.classes_,
        'Probability (%)': probs * 100
    }).sort_values('Probability (%)', ascending=False).reset_index(drop=True)

    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = prob_df.head(10)
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_n)))
        bars = ax.barh(top_n['Crop'][::-1], top_n['Probability (%)'][::-1], color=colors[::-1])
        ax.set_xlabel('Probability (%)', fontsize=12)
        ax.set_title('Top 10 Crop Probabilities', fontsize=14, fontweight='bold')
        for bar, prob in zip(bars, top_n['Probability (%)'][::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1f}%', va='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

    with right:
        st.dataframe(prob_df, use_container_width=True, height=400)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; padding:1rem;">
    <p>📌 <strong>Crop Recommendation System</strong> | Powered by Machine Learning &amp; Streamlit</p>
    <p style="font-size:0.8rem;">Built with ❤️ for agricultural decision support</p>
</div>
""", unsafe_allow_html=True)
