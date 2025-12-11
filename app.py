import nltk
# ensure nltk data is available on fresh servers (Render / Deploy)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)   # <--- fixes the Runtime LookupError you saw
nltk.download('stopwords', quiet=True)

import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    # remove non-alphanumeric
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    # remove stopwords & punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    # stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# --- load artifacts once ---
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
# -------------------------

st.title("SMS Spam Classification")
input_sms=st.text_input("Enter the message")

if st.button("Predict"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorise  (wrap inside a list!)
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

