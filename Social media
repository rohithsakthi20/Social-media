import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
# Load data
data = pd.read_csv("emotions.csv")
# Clean text
data['text'] = data['text'].str.lower()
data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', x))
data['text'] = data['text'].apply(lambda x: ' '.join([PorterStemmer().stem(word) for word in x.split() if word not in stopwords.words('english')]))
# Encode target
le = LabelEncoder()
data['label'] = le.fit_transform(data['emotion'])
# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['text']).toarray()
y = data['label']
