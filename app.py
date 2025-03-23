import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import re

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
    subjectivity = blob.sentiment.subjectivity
    return {"polarity": polarity, "sentiment": sentiment, "subjectivity": subjectivity}

def analyze_word_sentiment(text):
    words = word_tokenize(text)
    sentiment_data = [{"word": word, "polarity": TextBlob(word).sentiment.polarity} for word in words]
    return pd.DataFrame(sentiment_data)

def extract_aspects(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    aspects = [word for word, tag in tagged_words if tag in ["NN", "NNS", "NNP", "NNPS"]]
    aspect_sentiments = [{"Aspect": aspect, "Sentiment": TextBlob(aspect).sentiment.polarity} for aspect in aspects]
    return pd.DataFrame(aspect_sentiments)

def get_frequent_aspects(df, text_column):
    all_aspects = []
    for text in df[text_column].dropna():
        words = word_tokenize(str(text))
        tagged_words = pos_tag(words)
        aspects = [word for word, tag in tagged_words if tag in ["NN", "NNS", "NNP", "NNPS"]]
        all_aspects.extend(aspects)

    aspect_counts = Counter(all_aspects)
    aspect_df = pd.DataFrame(aspect_counts.items(), columns=["Aspect", "Frequency"]).sort_values(by="Frequency", ascending=False)
    return aspect_df

st.set_page_config(page_title="Sentiment Analysis Tool", page_icon="😊", layout="wide")

st.title("Sentiment Analysis Tool")
st.markdown("This app analyzes text sentiment and extracts important aspects.")

tab1, tab2, tab3 = st.tabs(["Single Text Analysis", "Batch Analysis", "About"])

with tab1:
    st.header("Analyze Text")
    text_input = st.text_area("Enter text to analyze:", height=150)

    if st.button("Analyze", key="analyze_single"):
        if text_input:
            processed_text = preprocess_text(text_input)
            results = analyze_sentiment(processed_text)

            st.subheader("Sentiment Analysis Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", results["sentiment"])
            col2.metric("Polarity", f"{results['polarity']:.2f}")
            col3.metric("Subjectivity", f"{results['subjectivity']:.2f}")

            word_sentiment_df = analyze_word_sentiment(processed_text)
            st.subheader("Word-Level Sentiment Analysis")
            st.dataframe(word_sentiment_df)
            fig = px.bar(word_sentiment_df, x="word", y="polarity", color="polarity", color_continuous_scale=["red", "gray", "green"])
            st.plotly_chart(fig)

            aspect_df = extract_aspects(processed_text)
            st.subheader("Aspect-Based Sentiment Analysis")
            st.dataframe(aspect_df)
            fig = px.pie(aspect_df, names="Aspect", values="Sentiment", title="Aspect Sentiment Distribution")
            st.plotly_chart(fig)
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Batch Analysis")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())

            text_column = st.selectbox("Select the column containing text:", df.columns)

            if st.button("Run Batch Analysis", key="analyze_batch"):
                with st.spinner("Analyzing data..."):
                    progress_bar = st.progress(0)
                    results = []
                    for i, row in enumerate(df[text_column]):
                        if pd.isna(row):
                            continue
                        processed_text = preprocess_text(str(row))
                        sentiment_results = analyze_sentiment(processed_text)
                        results.append(sentiment_results)
                        progress_bar.progress((i+1)/len(df))

                    results_df = pd.DataFrame(results)
                    output_df = pd.concat([df, results_df], axis=1)
                    st.subheader("Analysis Results")
                    st.dataframe(output_df)

                    st.subheader("Sentiment Distribution")
                    fig = px.pie(results_df, names='sentiment', title='Sentiment Distribution')
                    st.plotly_chart(fig)

                    fig = px.histogram(results_df, x='polarity', title='Polarity Distribution', color_discrete_sequence=['#3366CC'])
                    st.plotly_chart(fig)

                    fig = px.histogram(results_df, x='subjectivity', title='Subjectivity Distribution', color_discrete_sequence=['#FF9900'])
                    st.plotly_chart(fig)

                    aspect_df = get_frequent_aspects(df, text_column)
                    st.subheader("Frequent Aspects in Text Data")
                    st.dataframe(aspect_df)

                    fig = px.bar(aspect_df.head(10), x="Aspect", y="Frequency", title="Top 10 Frequent Aspects")
                    st.plotly_chart(fig)

                    csv = output_df.to_csv(index=False)
                    st.download_button("Download Results as CSV", csv, "sentiment_analysis_results.csv", "text/csv", key='download-csv')
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.header("About This Tool")
    st.markdown("""
    ## Sentiment Analysis Tool  
    - **Single Text Analysis**: Analyze sentiment of any text  
    - **Batch Analysis**: Process multiple texts from a CSV or Excel file  
    - **Word-Level Sentiment Analysis**  
    - **Aspect-Based Sentiment Analysis**  
    - **Frequent Aspect Analysis in Batch Processing**  
    """)

    st.info("Created for demonstration purposes. This tool provides general sentiment analysis but may not capture nuanced emotions or sarcasm.")

with st.sidebar:
    st.header("Options")
    theme = st.selectbox("Select theme:", ["Light", "Dark"], index=0)
    if theme == "Dark":
        st.markdown(""" <style>.stApp { background-color: #0E1117; color: white; } </style> """, unsafe_allow_html=True)

    st.subheader("Advanced Settings")
    st.checkbox("Show detailed token analysis", value=False)
    st.write("Sentiment Analysis Tool v1.1")
