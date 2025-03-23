# Sentiment Analysis Application

A Streamlit web application for analyzing sentiment in text data, either as single inputs or in batch mode through CSV/Excel files.

## Features

- Single text sentiment analysis
- Batch processing via CSV/Excel files
- Interactive visualizations (gauge charts, pie charts, histograms)
- Word cloud generation
- Text statistics (character count, word count, unique words)
- Results export as CSV

## Requirements

All requirements are listed in the `requirements.txt` file.

## Deployment

This application is deployed on Streamlit Cloud. You can access it at [your-app-url].

## Local Development

To run this application locally:

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## Usage

### Single Text Analysis
1. Enter or paste text in the text area
2. Click "Analyze" to view sentiment results

### Batch Analysis
1. Upload a CSV or Excel file containing a text column
2. Select the column to analyze
3. Click "Run Batch Analysis" to process all texts
4. Download results using the "Download Results as CSV" button
