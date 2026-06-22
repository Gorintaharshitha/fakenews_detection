import pandas as pd
import streamlit as st
import joblib
import os 
BASE_DIR =os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
st.title("Fake News Detection-web Application")
st.info(""" Instruction:\n1.Enter a complete news article.\n2.Avoid short Sentences.\n3.click the 'check news button' to seethe result.""")
user_input=st.text_area("Enter news article:")
if st.button("Check News"):
    if user_input.strip()=="":
        st.warning("please enter some news text.")
    else:

        input_data=vectorizer.transform([user_input])
        prediction=model.predict(input_data)
        if prediction[0]==1:
            st.success("Real News")
        else:
            st.error("Fake News")
st.markdown("**Developed by Harshitha**")