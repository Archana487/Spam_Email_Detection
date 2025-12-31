# streamlit_app.py
import streamlit as st
import joblib
import re

# ------------------------------
# Load saved model and TF-IDF vectorizer
# ------------------------------
model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ------------------------------
# Text cleaning function
# ------------------------------
def clean_text_simple(text):
    stop_words = set([
        'i','me','my','we','our','you','your','he','she','it','they','them','is',
        'are','was','were','be','been','am','the','a','an','and','or','in','on',
        'at','of','for','to','from','by','with','about','as','this','that','these',
        'those','then','there','here'
    ])
    text = text.lower()                      # lowercase
    text = re.sub(r'\d+', '', text)          # remove numbers
    text = re.sub(r'[^\w\s]', '', text)      # remove punctuation
    words = text.split()                      # split by spaces
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Spam Email Detector")

email_text = st.text_area("Enter the content of the email/message to check for spam:")


if st.button("Predict"):
    cleaned = clean_text_simple(email_text)
    vector = tfidf.transform([cleaned])
    result = model.predict(vector)
    
    if result[0] == 1:
        st.success("Spam 🚫")
    else:
        st.info("Not Spam ✅")
