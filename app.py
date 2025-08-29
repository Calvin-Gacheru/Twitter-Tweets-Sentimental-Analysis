# app.py
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("sentiment_model.pkl")

# Streamlit UI
st.title("Twitter Entity Sentiment Analysis")

st.write("Enter a tweet and select the entity to analyze sentiment.")

# Input fields
tweet_text = st.text_area("Tweet text")
entity_text = st.text_input("Entity (e.g., 'Apple', 'Google', 'Microsoft')")

if st.button("Predict Sentiment"):
    if tweet_text.strip() == "" or entity_text.strip() == "":
        st.warning("Please enter both tweet text and entity.")
    else:
        # Prepare input as dataframe (same structure as training data)
        input_df = pd.DataFrame([[tweet_text, entity_text]], columns=["text", "entity"])
        
        # Predict
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # Show result
        st.subheader("Predicted Sentiment:")
        st.success(prediction)

        st.write("Confidence:")
        st.write({cls: f"{p:.2%}" for cls, p in zip(model.classes_, proba)})
