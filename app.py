import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the trained model
with open('best_gradient_boosting_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Set up the Streamlit app with a title and header
st.title("Breast Cancer Prediction App")
st.markdown("""
    This app predicts whether a tumor is malignant or benign using **Gradient Boosting Classifier**.
    Please input the tumor characteristics to get a prediction and explore the model's decision process.
""")

# Interactive Feature Selection for Input
st.sidebar.header("Input Tumor Characteristics")
radius_mean = st.sidebar.slider('Radius Mean', 6.0, 30.0, 14.0)
texture_mean = st.sidebar.slider('Texture Mean', 9.0, 40.0, 18.0)
perimeter_mean = st.sidebar.slider('Perimeter Mean', 40.0, 200.0, 100.0)
area_mean = st.sidebar.slider('Area Mean', 150.0, 2500.0, 500.0)
concave_points_mean = st.sidebar.slider('Concave Points Mean', 0.0, 0.2, 0.05)
radius_worst = st.sidebar.slider('Radius Worst', 7.0, 40.0, 15.0)
perimeter_worst = st.sidebar.slider('Perimeter Worst', 50.0, 250.0, 125.0)
area_worst = st.sidebar.slider('Area Worst', 200.0, 4000.0, 1000.0)
concave_points_worst = st.sidebar.slider('Concave Points Worst', 0.0, 0.3, 0.1)

# Collect input data in a dictionary
input_data = {
    "radius_mean": radius_mean,
    "texture_mean": texture_mean,
    "perimeter_mean": perimeter_mean,
    "area_mean": area_mean,
    "concave points_mean": concave_points_mean,
    "radius_worst": radius_worst,
    "perimeter_worst": perimeter_worst,
    "area_worst": area_worst,
    "concave points_worst": concave_points_worst,
}

# Convert input data into a DataFrame
input_df = pd.DataFrame([input_data])

# Display the inputs as a DataFrame for feedback
st.subheader("Your Input Features")
st.write(input_df)

# Prediction button
if st.button("Predict"):
    # Prediction and confidence
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display prediction result
    if prediction[0] == 1:
        st.write("The tumor is **Malignant**. (Cancerous)")
    else:
        st.write("The tumor is **Benign**. (Non-cancerous)")

    # Display prediction confidence as a percentage
    confidence = prediction_proba[0][prediction[0]] * 100
    st.write(f"Prediction Confidence: **{confidence:.2f}%**")

    # Create a bar chart for the prediction probabilities
    st.subheader("Prediction Confidence")
    proba_data = pd.DataFrame({
        'Class': ['Benign', 'Malignant'],
        'Probability': prediction_proba[0]
    })
    st.bar_chart(proba_data.set_index('Class'))

# Model Explanation using Feature Importance
st.subheader("Model Feature Importance")
st.markdown("""
    The Gradient Boosting model assigns different levels of importance to the features when making predictions.
    Below is the plot that illustrates the relative importance of the selected features:
""")

# Get feature importances from the model
importances = model.feature_importances_
features = input_df.columns

# Sort the features by importance
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance in Breast Cancer Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
st.pyplot(plt)

# Additional Feedback Section
st.sidebar.markdown("""
---
**Note:** This model uses only a subset of the most important features that influence whether the tumor is benign or malignant. It has been trained to provide highly accurate results with minimal input complexity.
""")
