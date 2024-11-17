import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    sc = pickle.load(scaler_file)  # Loading as 'sc'

# Streamlit UI setup
st.title("Breast Cancer Prediction")
st.markdown("""
    This app predicts whether a breast tumor is **malignant** or **benign** based on cell characteristics.
""")

# Sidebar sliders for all features
user_inputs = {
    "radius_mean": st.sidebar.slider('Radius Mean', 5.0, 10.0, 30.0),
    "texture_mean": st.sidebar.slider('Texture Mean', 8.0, 20.0, 40.0),
    "perimeter_mean": st.sidebar.slider('Perimeter Mean', 40.0, 100.0, 200.0),
    "area_mean": st.sidebar.slider('Area Mean', 150.0, 1000.0, 2500.0),
    "smoothness_mean": st.sidebar.slider('Smoothness Mean', 0.05, 0.1, 0.2),
    "compactness_mean": st.sidebar.slider('Compactness Mean', 0.0, 0.1, 0.35),
    "concavity_mean": st.sidebar.slider('Concavity Mean', 0.0, 0.2, 0.4),
    "concave points_mean": st.sidebar.slider('Concave Points Mean', 0.0, 0.1, 0.2),
    "symmetry_mean": st.sidebar.slider('Symmetry Mean', 0.1, 0.2, 0.3),
    "fractal_dimension_mean": st.sidebar.slider('Fractal Dimension Mean', 0.04, 0.08, 0.1),
    "radius_se": st.sidebar.slider('Radius SE', 0.1, 1.0, 3.0),
    "texture_se": st.sidebar.slider('Texture SE', 0.2, 2.0, 5.0),
    "perimeter_se": st.sidebar.slider('Perimeter SE', 0.5, 10.0, 20.0),
    "area_se": st.sidebar.slider('Area SE', 5.0, 200.0, 400.0),
    "smoothness_se": st.sidebar.slider('Smoothness SE', 0.002, 0.02, 0.03),
    "compactness_se": st.sidebar.slider('Compactness SE', 0.002, 0.05, 0.10),
    "concavity_se": st.sidebar.slider('Concavity SE', 0.0, 0.2, 0.4),
    "concave points_se": st.sidebar.slider('Concave Points SE', 0.0, 0.01, 0.05),
    "symmetry_se": st.sidebar.slider('Symmetry SE', 0.01, 0.02, 0.08),
    "fractal_dimension_se": st.sidebar.slider('Fractal Dimension SE', 0.001, 0.01, 0.03),
    "radius_worst": st.sidebar.slider('Radius Worst', 5.0, 20.0, 35.0),
    "texture_worst": st.sidebar.slider('Texture Worst', 5.0, 20.0, 50.0),
    "perimeter_worst": st.sidebar.slider('Perimeter Worst', 40.0, 100.0, 250.0),
    "area_worst": st.sidebar.slider('Area Worst', 200.0, 2000.0, 4000.0),
    "smoothness_worst": st.sidebar.slider('Smoothness Worst', 0.1, 0.3, 0.15),
    "compactness_worst": st.sidebar.slider('Compactness Worst', 0.0, 1.2, 0.25),
    "concavity_worst": st.sidebar.slider('Concavity Worst', 0.0, 0.5, 1.0),
    "concave points_worst": st.sidebar.slider('Concave Points Worst', 0.0, 0.1, 0.3),
    "symmetry_worst": st.sidebar.slider('Symmetry Worst', 0.1, 0.4, 0.7),
    "fractal_dimension_worst": st.sidebar.slider('Fractal Dimension Worst', 0.05, 0.1, 0.25),
}

# Convert user input to DataFrame and scale it
input_data = pd.DataFrame([user_inputs])
input_data_scaled = sc.transform(input_data)

# Plot user inputs
st.subheader("Input Feature Overview")
fig, ax = plt.subplots(figsize=(10, 6))
pd.Series(user_inputs).plot(kind='bar', ax=ax, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Value")
plt.title("User-Selected Input Features")
st.pyplot(fig)

# Prediction and visualization
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    # Display prediction results
    if prediction[0] == 1:
        st.write("The tumor is **malignant**.")
    else:
        st.write("The tumor is **benign**.")
    confidence_score = prediction_proba[0][prediction[0]]
    st.write(f"Prediction confidence: {confidence_score:.2f}")

    
