import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# ---- Custom CSS for Colorful UI ----
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #667eea, #764ba2);
}

.main {
    background-color: #f7f9fc;
    padding: 30px;
    border-radius: 15px;
}

.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #4a148c;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #555;
}

.result-spam {
    background-color: #ffebee;
    color: #b71c1c;
    padding: 15px;
    border-radius: 10px;
    font-size: 18px;
    text-align: center;
}

.result-safe {
    background-color: #e8f5e9;
    color: #1b5e20;
    padding: 15px;
    border-radius: 10px;
    font-size: 18px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---- UI ----
st.markdown("<div class='title'>📧 Spam Email Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Machine Learning using Logistic Regression</div>", unsafe_allow_html=True)

st.markdown("### ✉️ Enter Email Message")
user_input = st.text_area(
    "Paste the email content here (not email ID)",
    height=180
)

# Predict button
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter an email message.")
    else:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)[0]

        if prediction == 1:
            st.markdown(
                "<div class='result-spam'>🚨 This email is SPAM</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-safe'>✅ This email is NOT SPAM</div>",
                unsafe_allow_html=True
            )

# Examples
st.markdown("### 📌 Example Messages")
st.markdown("""
**🚨 Spam Example:**  
Congratulations! You have won a free gift card. Click now to claim.

**✅ Normal Example:**  
Hi, the meeting is scheduled for tomorrow at 10 AM.
""")

# Footer
st.markdown("---")
st.caption("Developed by Archana | Spam Email Detection Project")

