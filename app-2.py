import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and scaler
with open('best_gradient_boosting_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

# Streamlit UI setup
st.set_page_config(page_title="Breast Cancer Prediction App", layout="centered")
st.title("Breast Cancer Prediction App")
st.markdown("""
    This app predicts whether a breast tumor is **malignant** or **benign** based on various tumor cell characteristics.
""")

# Sidebar for simplified layout
st.sidebar.header("Input Features")

# Mixed input methods for key features
user_inputs = {
    "radius_mean": st.sidebar.number_input('Radius Mean', min_value=6.0, max_value=30.0, value=14.0, step=0.1),
    "texture_mean": st.sidebar.slider('Texture Mean', 9.0, 40.0, 18.0),
    "perimeter_mean": st.sidebar.slider('Perimeter Mean', 40.0, 200.0, 100.0),
    "area_mean": st.sidebar.number_input('Area Mean', min_value=150.0, max_value=2500.0, value=500.0, step=10.0),
    "smoothness_mean": st.sidebar.slider('Smoothness Mean', 0.05, 0.2, 0.1),
    "compactness_mean": st.sidebar.number_input('Compactness Mean', min_value=0.0, max_value=0.35, value=0.1, step=0.01),
    "concavity_mean": st.sidebar.number_input('Concavity Mean', min_value=0.0, max_value=0.5, value=0.1, step=0.01),
    "concave points_mean": st.sidebar.slider('Concave Points Mean', 0.0, 0.2, 0.05),
    "symmetry_mean": st.sidebar.number_input('Symmetry Mean', min_value=0.1, max_value=0.3, value=0.2, step=0.01),
    "fractal_dimension_mean": st.sidebar.slider('Fractal Dimension Mean', 0.05, 0.2, 0.1),
}

# Convert user input to DataFrame and scale it
input_data = pd.DataFrame([user_inputs])
input_data_scaled = sc.transform(input_data)

# Display a summary of the user inputs
st.subheader("Input Summary")
st.write(input_data)

# Radar chart to show selected features relative to their min and max values
st.subheader("Feature Visualization")
categories = list(user_inputs.keys())
values = list(user_inputs.values())
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(np.linspace(0, 2 * np.pi, len(categories)), values, color='skyblue', alpha=0.4)
ax.plot(np.linspace(0, 2 * np.pi, len(categories)), values, color='blue', linewidth=2)
ax.set_xticks(np.linspace(0, 2 * np.pi, len(categories)))
ax.set_xticklabels(categories, fontsize=9)
plt.title("Selected Feature Profile")
st.pyplot(fig)

# Prediction and confidence visualization
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    # Display prediction result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.write("The tumor is **malignant**.")
    else:
        st.write("The tumor is **benign**.")

    # Display confidence in a horizontal bar chart
    st.subheader("Prediction Confidence")
    confidence_labels = ["Benign", "Malignant"]
    confidence_values = prediction_proba[0]
    fig, ax = plt.subplots(figsize=(6, 1.5))
    sns.barplot(x=confidence_values, y=confidence_labels, palette="coolwarm", orient="h", ax=ax)
    plt.xlabel("Probability")
    plt.title("Confidence in Prediction")
    st.pyplot(fig)
