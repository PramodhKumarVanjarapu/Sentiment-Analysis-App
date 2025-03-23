import nltk
import spacy
import subprocess

def download_models():
    # Download NLTK data
    nltk.download('punkt')
    
    # Download spaCy model
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

if __name__ == "__main__":
    download_models()
