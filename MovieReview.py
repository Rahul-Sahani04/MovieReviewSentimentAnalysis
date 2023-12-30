# %% [markdown]
# ### Import necessary libraries

# %%
# Import necessary libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# %%
# Function to scrape IMDb movie reviews
# %writefile MovieReview.py
def scrape_imdb_reviews(movie_url):
    response = requests.get(movie_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extracting review text
    reviews = [{'text': review_div.get_text(strip=True)} for review_div in soup.find_all('a', class_='title')]
    
    return reviews

# Function for text cleaning
def clean_text(text):
    cleaned_text = re.sub(r"[^\w\s]", "", text.lower())  # Convert to lowercase and remove special characters
    return cleaned_text

# Function for tokenization
def tokenize_text(text):
    return nltk.word_tokenize(text)

# Function for removing stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token.lower() not in stop_words]

# Function for lemmatization
def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

if __name__ == "__main__":
    # %%
    # IMDb movie URL (replace with the actual movie URL)
    IMDB_Code = "tt4633694"
    movie_url = f'https://www.imdb.com/title/{IMDB_Code}/reviews'

    # Scrape IMDb movie reviews
    movie_reviews = scrape_imdb_reviews(movie_url)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(movie_reviews)
    df.head()

    # Data Preprocessing
    df["text_cleaned"] = df["text"].apply(clean_text)
    df["tokens"] = df["text_cleaned"].apply(tokenize_text)
    df["tokens"] = df["tokens"].apply(remove_stopwords)
    df["tokens"] = df["tokens"].apply(lemmatize_text)

# %%
# Sentiment Analysis using VADER
# nltk.download('vader_lexicon')

    vader = SentimentIntensityAnalyzer()
    df['compound'] = df['text_cleaned'].apply(lambda x: vader.polarity_scores(x)['compound'])

    # Classify sentiments based on compound score
    df['predicted_sentiment'] = df['compound'].apply(lambda x: 'positive' if x >= 0 else 'negative')

    df.head()  # Display the first few rows of the DataFrame

    # %%
    # Model Evaluation
    accuracy = accuracy_score(df['predicted_sentiment'], df['predicted_sentiment'])
    print(f'Accuracy: {accuracy:.2f}')

    # Display classification report
    print('\nClassification Report:')
    print(classification_report(df['predicted_sentiment'], df['predicted_sentiment']))

    # %%
    df['predicted_sentiment'].head()

    # %%
    # Visualize the distribution of predicted sentiments
    plt.figure(figsize=(6, 4))
    df['predicted_sentiment'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Distribution of Predicted Sentiments in IMDb Reviews')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Count')
    plt.show()

# %%



