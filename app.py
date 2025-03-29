import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer



def predict_sentiment(text, model, vectorizer, stopwords):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in set(stopwords)]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)

    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"


def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; margin: 10px 0; border-radius: 5px;">
        <h5 style="color: white;">Sentiment: {sentiment}</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

def main():
    st.title("Twitter Sentiment Analysis")

    stopwords = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    
    text_input = st.text_area("Enter text for sentiment analysis")
    if st.button("Analyze"):
        sentiment = predict_sentiment(text_input, model, vectorizer, stopwords)
        st.write(f"Sentiment: {sentiment}")


if __name__ == "__main__":
    main()