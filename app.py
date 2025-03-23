import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from textblob import TextBlob
import re
import spacy
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load spaCy model
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # If model not found, download it
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

# Load ABSA model
@st.cache_resource
def load_absa_model():
    model_name = "yangheng/deberta-v3-base-absa-v1.1"
    return pipeline("text-classification", model=model_name)

# Initialize models
nlp = load_nlp_model()
STOPWORDS = set(["the", "a", "an", "is", "was", "were", "it", "this", "that", "of", "to", "for", "on", "with", "as", "by", "at", "in", "and", "but", "or"])

# Define ABSA functions
def predict_aspects(text):
    doc = nlp(text)
    aspects = []

    for chunk in doc.noun_chunks:
        # Include nouns and compound modifiers
        aspect_tokens = []
        for token in chunk:
            if (token.text.lower() not in STOPWORDS and
                (token.pos_ in ["NOUN", "PROPN"] or
                 (token.dep_ == "compound" and token.head.pos_ in ["NOUN", "PROPN"])) and
                token.dep_ not in ["det", "poss"]):
                aspect_tokens.append(token.text)

        aspect = " ".join(aspect_tokens).strip()
        if aspect:
            aspects.append(aspect)

    return sorted(list(set(filter(None, aspects))))

def classify_sentiment_absa(text, aspect_terms, absa_classifier):
    aspect_sentiments = []
    for aspect in aspect_terms:
        input_text = f"[CLS] {text} [SEP] {aspect} [SEP]"
        result = absa_classifier(input_text)
        sentiment = result[0]['label'].lower()
        confidence = result[0]['score']
        aspect_sentiments.append((aspect, sentiment, confidence))
    return aspect_sentiments

def get_aspect_sentiments(text, absa_classifier):
    aspect_terms = predict_aspects(text)
    if aspect_terms:
        aspect_sentiments = classify_sentiment_absa(text, aspect_terms, absa_classifier)
    else:
        aspect_sentiments = []
    return aspect_sentiments

def preprocess_text(text):
    """Clean and preprocess the text."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob."""
    blob = TextBlob(text)
    
    # Get polarity score (-1 to 1)
    polarity = blob.sentiment.polarity
    
    # Determine sentiment label
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Get subjectivity score (0 to 1)
    subjectivity = blob.sentiment.subjectivity
    
    return {
        "polarity": polarity,
        "sentiment": sentiment,
        "subjectivity": subjectivity
    }

def generate_word_cloud(text, title="Word Cloud"):
    """Generate a word cloud from text."""
    # Tokenize
    word_tokens = word_tokenize(text)
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white', 
                          max_words=150, 
                          contour_width=3, 
                          contour_color='steelblue').generate(' '.join(word_tokens))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    return fig

# Set page config
st.set_page_config(
    page_title="Advanced Sentiment Analysis Tool",
    page_icon="😊",
    layout="wide"
)

# App title and introduction
st.title("Advanced Sentiment Analysis Tool")
st.markdown("""
    This application analyzes sentiment of text using both traditional methods and 
    Aspect-Based Sentiment Analysis (ABSA). ABSA breaks down text to identify specific 
    aspects mentioned and their associated sentiment.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Single Text Analysis", "Batch Analysis", "About"])

with tab1:
    st.header("Analyze Text")
    text_input = st.text_area("Enter text to analyze:", height=150)
    
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.radio(
            "Analysis type",
            ["Basic Sentiment Analysis", "Aspect-Based Sentiment Analysis (ABSA)", "Both"]
        )
    
    if st.button("Analyze", key="analyze_single"):
        if text_input:
            with st.spinner("Analyzing text..."):
                # Preprocess text
                processed_text = preprocess_text(text_input)
                
                # Basic sentiment analysis
                if analysis_type in ["Basic Sentiment Analysis", "Both"]:
                    results = analyze_sentiment(processed_text)
                    
                    # Display results
                    st.subheader("Overall Sentiment Analysis")
                    
                    # Create three columns
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    # Display metrics
                    metric_col1.metric("Sentiment", results["sentiment"])
                    metric_col2.metric("Polarity", f"{results['polarity']:.2f}")
                    metric_col3.metric("Subjectivity", f"{results['subjectivity']:.2f}")
                    
                    # Display gauge chart for polarity
                    fig = px.gauge(
                        value=results["polarity"],
                        range_color=[-1, 1],
                        color_discrete_sequence=["red", "gray", "green"],
                        title="Sentiment Polarity (-1: Very Negative to 1: Very Positive)"
                    )
                    st.plotly_chart(fig)
                
                # Aspect-based sentiment analysis
                if analysis_type in ["Aspect-Based Sentiment Analysis (ABSA)", "Both"]:
                    st.subheader("Aspect-Based Sentiment Analysis")
                    
                    # Load ABSA model if not loaded
                    with st.spinner("Loading ABSA model..."):
                        absa_classifier = load_absa_model()
                    
                    # Get aspect sentiments
                    aspect_sentiments = get_aspect_sentiments(text_input, absa_classifier)
                    
                    if aspect_sentiments:
                        # Create a DataFrame for the results
                        aspects = [a[0] for a in aspect_sentiments]
                        sentiments = [a[1] for a in aspect_sentiments]
                        confidences = [a[2] for a in aspect_sentiments]
                        
                        df_aspects = pd.DataFrame({
                            "Aspect": aspects,
                            "Sentiment": sentiments,
                            "Confidence": confidences
                        })
                        
                        # Display results in a table
                        st.dataframe(df_aspects)
                        
                        # Create a horizontal bar chart for aspects
                        fig = px.bar(
                            df_aspects,
                            y="Aspect",
                            x="Confidence",
                            color="Sentiment",
                            color_discrete_map={"positive": "green", "negative": "red", "neutral": "gray"},
                            title="Aspect Sentiment Analysis",
                            orientation="h"
                        )
                        st.plotly_chart(fig)
                    else:
                        st.info("No specific aspects were identified in the text.")
                
                # Generate word cloud
                st.subheader("Word Cloud")
                wordcloud_fig = generate_word_cloud(processed_text)
                st.pyplot(wordcloud_fig)
                
                # Display token analysis
                st.subheader("Text Analysis")
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
    
    # File uploader
    st.subheader("Upload a CSV or Excel file")
    uploaded_file = st.file_uploader("The file should contain a column with text to analyze", 
                                     type=["csv", "xlsx", "xls"])
    
    batch_analysis_type = st.radio(
        "Batch analysis type",
        ["Basic Sentiment Analysis", "Aspect-Based Sentiment Analysis (ABSA)"]
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display the dataframe
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Column selection
            text_column = st.selectbox("Select the column containing text for analysis:", df.columns)
            
            if st.button("Run Batch Analysis", key="analyze_batch"):
                with st.spinner("Analyzing data..."):
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Basic sentiment analysis
                    if batch_analysis_type == "Basic Sentiment Analysis":
                        # Initialize results
                        results = []
                        
                        # Process each row
                        for i, row in enumerate(df[text_column]):
                            # Skip NaN values
                            if pd.isna(row):
                                continue
                                
                            # Preprocess and analyze text
                            processed_text = preprocess_text(str(row))
                            sentiment_results = analyze_sentiment(processed_text)
                            results.append(sentiment_results)
                            
                            # Update progress
                            progress_bar.progress((i+1)/len(df))
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(results)
                        
                        # Combine with original dataframe
                        output_df = pd.concat([df, results_df], axis=1)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        st.dataframe(output_df)
                        
                        # Create visualizations
                        st.subheader("Sentiment Distribution")
                        fig = px.pie(results_df, names='sentiment', title='Sentiment Distribution')
                        st.plotly_chart(fig)
                        
                        # Polarity histogram
                        fig = px.histogram(results_df, x='polarity', 
                                           title='Polarity Distribution',
                                           color_discrete_sequence=['#3366CC'])
                        st.plotly_chart(fig)
                    
                    # Aspect-based sentiment analysis
                    else:
                        # Load ABSA model
                        with st.spinner("Loading ABSA model..."):
                            absa_classifier = load_absa_model()
                        
                        all_aspects = []
                        all_texts = []
                        
                        # First pass: collect all aspects
                        for i, row in enumerate(df[text_column]):
                            if pd.isna(row):
                                continue
                                
                            text = str(row)
                            aspect_sentiments = get_aspect_sentiments(text, absa_classifier)
                            
                            for aspect, sentiment, confidence in aspect_sentiments:
                                all_aspects.append({
                                    "text_id": i,
                                    "aspect": aspect,
                                    "sentiment": sentiment,
                                    "confidence": confidence
                                })
                            
                            all_texts.append({
                                "text_id": i,
                                "text": text
                            })
                            
                            # Update progress
                            progress_bar.progress((i+1)/len(df))
                        
                        # Create aspects dataframe
                        aspects_df = pd.DataFrame(all_aspects)
                        texts_df = pd.DataFrame(all_texts)
                        
                        if not aspects_df.empty:
                            # Display results
                            st.subheader("Aspect-Based Analysis Results")
                            
                            # Join with original text
                            result_with_text = aspects_df.merge(texts_df, on="text_id")
                            
                            # Display aspect results
                            st.dataframe(result_with_text)
                            
                            # Aspect sentiment distribution
                            st.subheader("Aspect Sentiment Distribution")
                            fig = px.pie(aspects_df, names='sentiment', title='Aspect Sentiment Distribution')
                            st.plotly_chart(fig)
                            
                            # Top aspects
                            top_aspects = aspects_df["aspect"].value_counts().reset_index()
                            top_aspects.columns = ["aspect", "count"]
                            top_aspects = top_aspects.head(10)
                            
                            fig = px.bar(top_aspects, y="aspect", x="count", 
                                         title="Top 10 Aspects Mentioned",
                                         orientation="h")
                            st.plotly_chart(fig)
                            
                            # Download option
                            csv = result_with_text.to_csv(index=False)
                            st.download_button(
                                "Download Results as CSV",
                                csv,
                                "aspect_sentiment_results.csv",
                                "text/csv",
                                key='download-csv'
                            )
                        else:
                            st.warning("No aspects were identified in any of the texts.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

with tab3:
    st.header("About This Tool")
    st.markdown("""
    ## Advanced Sentiment Analysis Tool
    
    This application combines traditional sentiment analysis with Aspect-Based Sentiment Analysis (ABSA) to provide deeper insights into text sentiment.
    
    ### Features:
    - **Basic Sentiment Analysis**: Overall sentiment of the entire text
    - **Aspect-Based Sentiment Analysis**: Identifies specific aspects mentioned in the text and their associated sentiment
    - **Batch Analysis**: Process multiple texts from a CSV or Excel file
    - **Visualization**: View sentiment distribution, aspect-level sentiment, and more
    - **Word Cloud**: Visualize the most frequent words in your text
    
    ### How It Works:
    
    #### Basic Sentiment Analysis
    Uses TextBlob to calculate:
    - **Polarity**: A float value ranging from -1 to 1 (negative to positive)
    - **Subjectivity**: A float value ranging from 0 to 1 (objective to subjective)
    
    #### Aspect-Based Sentiment Analysis (ABSA)
    Uses a combination of SpaCy for aspect extraction and a transformer model for sentiment classification:
    1. Extracts noun phrases as potential aspects
    2. Filters aspects based on linguistic rules
    3. Classifies the sentiment associated with each aspect
    
    ### Tips for Best Results:
    - Provide detailed, specific feedback in your text for better aspect extraction
    - For batch processing, ensure your texts have enough context
    - ABSA works best on review-style text that mentions specific features or aspects
    """)
    
    st.info("This tool demonstrates how to combine multiple NLP techniques for deeper text analysis. The ABSA model may take a moment to load the first time it's used.")

# Add sidebar with additional options
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
    
    st.markdown("---")
    st.write("Advanced Sentiment Analysis Tool v2.0")
