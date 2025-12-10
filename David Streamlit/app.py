import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model & TF-IDF
model = joblib.load("mental_health_xgb_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Title
st.set_page_config(page_title="Mental Health Disorder Prediction", layout="wide")
st.title(" Mental Health Disorder Prediction App")
st.write("Enter a text statement and the system predicts the associated mental health condition.")

# Input Text Box
user_input = st.text_area(" Enter your text here:")

# Label Mapping
label_map = {
    0: 'Stress',
    1: 'Depression',
    2: 'Bipolar disorder',
    3: 'Personality disorder',
    4: 'Anxiety'
}

# Predict Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        # Convert input into tf-idf format
        transformed_input = tfidf.transform([user_input])
        
        # Predict class
        prediction = model.predict(transformed_input)[0]
        prediction_label = label_map[prediction]

        # Predict probabilities
        probabilities = model.predict_proba(transformed_input)[0]

        # Display result
        st.success(f"### Predicted Disorder: **{prediction_label}**")

        # Show probability scores
        st.subheader(" Prediction Confidence:")
        prob_df = pd.DataFrame({"Disorder": list(label_map.values()),"Probability": probabilities})

        st.bar_chart(prob_df.set_index("Disorder"))

st.markdown("---")
st.caption("Deployment Demo • Mental Health Classification using Machine Learning")
