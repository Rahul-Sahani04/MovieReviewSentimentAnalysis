import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import os
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report

nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Function to scrape IMDb movie reviews
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

# Function to perform sentiment analysis
def perform_sentiment_analysis(movie_code):
    # IMDb movie URL
    movie_url = f'https://www.imdb.com/title/{movie_code}/reviews'

    # Scrape IMDb movie reviews
    movie_reviews = scrape_imdb_reviews(movie_url)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(movie_reviews)

    # Data Preprocessing
    df["text_cleaned"] = df["text"].apply(clean_text)
    df["tokens"] = df["text_cleaned"].apply(tokenize_text)
    df["tokens"] = df["tokens"].apply(remove_stopwords)
    df["tokens"] = df["tokens"].apply(lemmatize_text)

    # Sentiment Analysis using VADER
    vader = SentimentIntensityAnalyzer()
    df['compound'] = df['text_cleaned'].apply(lambda x: vader.polarity_scores(x)['compound'])

    # Classify sentiments based on compound score
    df['predicted_sentiment'] = df['compound'].apply(lambda x: 'positive' if x >= 0 else 'negative')

    # Save results to CSV
    save_path = "results"
    df.to_csv(os.path.join(save_path, f"movie_reviews_with_sentiment_for_{movie_code}.csv"), index=False)

    # Visualize the distribution of predicted sentiments
    colors = ["green" if col.lower() == "positive" else "red" for col in df["predicted_sentiment"].value_counts().index]
    fig, ax = plt.subplots(figsize=(6, 4))
    df['predicted_sentiment'].value_counts().plot(kind='bar', color=colors, ax=ax)
    ax.set_title(f'Distribution of Predicted Sentiments in IMDb Reviews ({movie_code})')
    ax.set_xlabel('Predicted Sentiment')
    ax.set_ylabel('Count')
    plt.tight_layout()

    # Save the plot to a file
    plot_file_path = os.path.join(save_path, f"sentiment_distribution_for_{movie_code}.png")
    plt.savefig(plot_file_path)

    # Display a message indicating positive sentiment
    more_counts = df['predicted_sentiment'].value_counts().index 
    positive_message = "The movie has positive reviews! You should consider watching it."
    negative_message = "The sentiment analysis did not identify a clear positive sentiment in the reviews."

    st.title("IMDb Movie Reviews Sentiment Analysis")
    st.image(plot_file_path, caption=f'Distribution of Predicted Sentiments in IMDb Reviews ({movie_code})', use_column_width=True)
    st.subheader("Sentiment Analysis Results:")
    st.write(df[['text', 'predicted_sentiment']])
    st.subheader("Sentiment Overview:")
    if "positive" == more_counts[0]:
        st.success(positive_message)
    else:
        st.warning(negative_message)


# Streamlit app
st.title("IMDb Movie Reviews Sentiment Analysis")

st.image("HowToGetIMDBCode.png", caption="How To Get IMDB Code", use_column_width=True)
# Input field for IMDb code
movie_code = st.text_input("Enter IMDb Code (e.g., tt12915716):")

# Button to trigger sentiment analysis
if st.button("Perform Sentiment Analysis"):
    if movie_code:
        perform_sentiment_analysis(movie_code)
    else:
        st.warning("Please enter a valid IMDb Code.")
