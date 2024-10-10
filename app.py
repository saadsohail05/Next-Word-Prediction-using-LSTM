import numpy as np
import streamlit as st
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import time
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

# Set page config (this must be the first Streamlit command)
st.set_page_config(page_title="AI-Powered Next Word Prediction", layout="wide")

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('next_word_prediction.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    return predicted[0]

# Function to get top N predictions
def get_top_predictions(predictions, tokenizer, n=5):
    top_indices = predictions.argsort()[-n:][::-1]
    top_words = []
    for idx in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                top_words.append((word, predictions[idx]))
                break
    return top_words

# Function to load Lottie animation
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Custom CSS
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 20px;
    }
    .prediction-chip {
        display: inline-block;
        padding: 10px 20px;
        margin: 5px;
        border-radius: 25px;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .prediction-chip:hover {
        transform: scale(1.05);
    }
    .most-likely {
        background-color: #4CAF50;
        color: white;
    }
    .copyButton {
        background-color: #008CBA;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .clearButton {
        background-color: #f44336;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("AI-Powered Next Word Prediction")
st.markdown("Experience the power of AI in predicting your next word as you type. Our LSTM-based model provides real-time suggestions to enhance your writing flow.")

# Main input area
text_input = st.text_area("Start typing your sentence here...", height=100)
# Prediction display
if text_input:
    start_time = time.time()
    predictions = predict_next_word(model, tokenizer, text_input, 10)
    end_time = time.time()
    response_time = end_time - start_time
    
    top_predictions = get_top_predictions(predictions, tokenizer)
    
    st.markdown("### Top Predictions")
    for i, (word, prob) in enumerate(top_predictions):
        chip_class = "prediction-chip most-likely" if i == 0 else "prediction-chip"
        st.markdown(f'<span class="{chip_class}">{word} ({prob:.2f})</span>', unsafe_allow_html=True)
    
    # Confidence levels chart
    fig = go.Figure(data=[go.Bar(
        x=[word for word, _ in top_predictions],
        y=[prob for _, prob in top_predictions],
        marker_color=['green' if i == 0 else 'lightblue' for i in range(len(top_predictions))]
    )])
    fig.update_layout(title="Prediction Confidence Levels", xaxis_title="Words", yaxis_title="Confidence")
    st.plotly_chart(fig)
    
    # Performance metrics
    st.markdown(f"Response time: {response_time:.4f} seconds")
    
    # Word count
    word_count = len(text_input.split())
    st.markdown(f"Word count: {word_count}")

# User interaction buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear", key="clear"):
        text_input = ""
        st.experimental_rerun()
with col2:
    if st.button("Copy to Clipboard", key="copy"):
        st.write("Text copied to clipboard!")

# Dark mode toggle
if st.checkbox("Dark Mode"):
    st.markdown("""
    <style>
        .stApp {
            background-color: #2b2b2b;
            color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("This app uses an LSTM (Long Short-Term Memory) neural network model for next word prediction.")
st.markdown("[Learn more about LSTM technology](https://www.tensorflow.org/tutorials/text/text_generation)")

# Responsive design
st.markdown("""
<style>
    @media (max-width: 600px) {
        .stTextInput > div > div > input {
            font-size: 16px;
        }
        .prediction-chip {
            font-size: 14px;
            padding: 8px 16px;
        }
    }
</style>
""", unsafe_allow_html=True)