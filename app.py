import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import re

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_text(text):
    """Clean and preprocess the text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    subjectivity = blob.sentiment.subjectivity
    return {
        "polarity": polarity,
        "sentiment": sentiment,
        "subjectivity": subjectivity
    }

def generate_word_cloud(text, title="Word Cloud"):
    """Generate a word cloud from text."""
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white', 
                          max_words=150, 
                          contour_width=3, 
                          contour_color='steelblue').generate(' '.join(filtered_text))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    return fig

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😊",
    layout="wide"
)

# App title and introduction
st.title("Sentiment Analysis Tool")
st.markdown("""
    This application analyzes the sentiment of text using natural language processing.
    Enter text in the box below or upload a file to get started.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Single Text Analysis", "Batch Analysis", "About"])

with tab1:
    st.header("Analyze Text")
    text_input = st.text_area("Enter text to analyze:", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze", key="analyze_single"):
            if text_input:
                with st.spinner("Analyzing text..."):
                    processed_text = preprocess_text(text_input)
                    results = analyze_sentiment(processed_text)
                    st.subheader("Sentiment Analysis Results")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric("Sentiment", results["sentiment"])
                    metric_col2.metric("Polarity", f"{results['polarity']:.2f}")
                    metric_col3.metric("Subjectivity", f"{results['subjectivity']:.2f}")

                    # **Fixed Gauge Chart**
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=results["polarity"],
                        title={'text': "Sentiment Polarity (-1: Very Negative to 1: Very Positive)"},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "gray"},
                            'steps': [
                                {'range': [-1, -0.1], 'color': "red"},
                                {'range': [-0.1, 0.1], 'color': "gray"},
                                {'range': [0.1, 1], 'color': "green"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig)

                    st.subheader("Word Cloud")
                    wordcloud_fig = generate_word_cloud(processed_text)
                    st.pyplot(wordcloud_fig)

                    tokens = word_tokenize(processed_text)
                    st.info(f"""
                        **Text Statistics:**
                        - Character count: {len(processed_text)}
                        - Word count: {len(tokens)}
                        - Unique words: {len(set(tokens))}
                    """)
            else:
                st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Batch Analysis")
    st.subheader("Upload a CSV or Excel file")
    uploaded_file = st.file_uploader("The file should contain a column with text to analyze", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            text_column = st.selectbox("Select the column containing text for analysis:", df.columns)
            
            if st.button("Run Batch Analysis", key="analyze_batch"):
                with st.spinner("Analyzing data..."):
                    progress_bar = st.progress(0)
                    results = []
                    for i, row in enumerate(df[text_column].dropna()):
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

                    csv = output_df.to_csv(index=False)
                    st.download_button("Download Results as CSV", csv, "sentiment_analysis_results.csv", "text/csv", key='download-csv')
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

with tab3:
    st.header("About This Tool")
    st.markdown("""
    ## Sentiment Analysis Tool
    This application uses NLP techniques to analyze text sentiment.
    
    ### Features:
    - **Single Text Analysis**
    - **Batch Analysis**
    - **Visualizations**
    - **Word Cloud**
    
    ### Sentiment Analysis Metrics:
    - **Polarity** (-1 to 1)
    - **Subjectivity** (0 to 1)
    """)

# Sidebar
with st.sidebar:
    st.header("Options")
    st.subheader("Theme")
    theme = st.selectbox("Select theme:", ["Light", "Dark"], index=0)
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    st.subheader("Advanced Settings")
    st.checkbox("Show detailed token analysis", value=False)
    st.markdown("---")
    st.write("Sentiment Analysis Tool v1.0")
