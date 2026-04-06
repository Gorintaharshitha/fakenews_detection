import pandas as pd
import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
st.title("Fake News Detection App")
user_input=st.text_area("Enter news..")
if st.button("Click News"):
    input_data=vectorizer.transform([user_input])
    prediction=model.predict(input_data)
    if prediction[0]==0:
        st.error("Fake News")
    else:
        st.success("Real News")
