import streamlit as st
import pickle

# Load the model and vectorizer
with open('classifier_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title("Medicine Predictor")

# Input fields
description = st.text_input("Enter Description:")
reason = st.text_input("Enter Reason:")

# Make predictions
if st.button("Predict Drug"):
    if not description or not reason:
        st.warning("Please enter both Description and Reason.")
    else:
        text = reason + ' ' + description
        text_vectorized = vectorizer.transform([text])
        prediction = classifier.predict(text_vectorized)
        st.success(f"Predicted Drug Name: {prediction[0]}")
