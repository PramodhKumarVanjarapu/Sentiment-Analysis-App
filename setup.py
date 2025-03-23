import nltk
import streamlit as st

def download_nltk_data():
    # Download NLTK data that's needed for the app
    nltk.download('punkt')
    nltk.download('stopwords')

# Run the function
if __name__ == "__main__":
    download_nltk_data()
