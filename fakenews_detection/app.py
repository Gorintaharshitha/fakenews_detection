import pandas as pd
import streamlit as st
import joblib
import os 
BASE_DIR =os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
st.title("Fake News Detection-web Application")
st.info(""" 
Instructions:
        
1.Enter a complete news article.
        
2.Avoid  very short Sentences.
        
 3.click the 'check news button' to see the result.
""")
user_input=st.text_area("Enter news article:")
if st.button("Check News"):
    if user_input.strip()=="":
        st.warning("please enter anews article.")
    else:

        input_data=vectorizer.transform([user_input])
        prediction=model.predict(input_data)
        if prediction[0]==1:
            st.success("Real News")
        else:
            st.error("Fake News")
st.caption("Note:Predictions are based on machine learning model trained on historical news data and may not verify current news events.")
st.markdown("**Developed by Harshitha**")