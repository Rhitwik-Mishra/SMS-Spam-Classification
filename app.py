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

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
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

