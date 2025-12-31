# Spam Email Detection Project

## Project Overview
This project builds a machine learning model to automatically detect whether an email is spam or not using Logistic Regression.  
It helps improve email security and reduces unwanted spam messages.

## Folder Structure
Spam_Email_Project/
├── spam.csv
├── Spam_Email_Detection.ipynb
├── spam_model.pkl
├── tfidf_vectorizer.pkl
├── streamlit_app.py
├── README.md
└── report.pdf

markdown
Copy code

## How to Run

### 1. Notebook
1. Open `Spam_Email_Detection.ipynb` in Jupyter Notebook.
2. Run all cells from top to bottom.
3. The notebook will:
   - Clean the data
   - Convert text to numerical features (TF-IDF)
   - Train Logistic Regression
   - Predict spam
   - Display accuracy and confusion matrix
4. Optional: Compare performance with Naive Bayes or SVM.

### 2. Streamlit App
1. Ensure `spam_model.pkl` and `tfidf_vectorizer.pkl` are in the same folder as `streamlit_app.py`.
2. Open terminal in the project folder.
3. Run:
streamlit run streamlit_app.py

markdown
Copy code
4. Type the **email content** (not your email ID) into the input box.
5. Click **Predict** to see **Spam** or **Not Spam**.

## Libraries Required
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- joblib
- streamlit

## Notes
- TF-IDF vectorization is used to convert text into numerical features.  
- Logistic Regression identifies spam based on important words.  
- Optional: You can test Naive Bayes or SVM for comparison.  
- Streamlit app allows you to enter new emails interactively and check for spam.
