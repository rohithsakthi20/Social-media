# Social-media ```
import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def textblob_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def detect_emotion(text):
    if 'happy' in text or 'joy' in text:
        return 'Happy'
    elif 'sad' in text or 'depress' in text:
        return 'Sad'
    elif 'angry' in text or 'mad' in text:
        return 'Angry'
    else:
        return 'Neutral'

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_vec, y_train)
    return clf, vectorizer

def evaluate_model(clf, vectorizer, X_test, y_test):
    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

def main():
    # Load dataset
    df = load_data('social_media_data.csv')

    # Preprocess text data
    df['text'] = df['text'].apply(preprocess_text)

    # Sentiment analysis using NLTK's VADER
    df['sentiment'] = df['text'].apply(sentiment_analysis)

    # Sentiment analysis using TextBlob
    df['textblob_sentiment'] = df['text'].apply(textblob_sentiment)

    # Emotion detection
    df['emotion'] = df['text'].apply(detect_emotion)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)

    # Train a random forest classifier
    clf, vectorizer = train_model(X_train, y_train)

    # Evaluate model performance
    evaluate_model(clf, vectorizer, X_test, y_test)

if __name__ == "__main__":
    main()
```
