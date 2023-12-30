# %%

# pip3 install scikit-learn

import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
import random

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
# Download the twitter_samples dataset
# nltk.download('twitter_samples')

# Import twitter_samples dataset
from nltk.corpus import twitter_samples

# Begin text vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_df=0.8)

# Initialize the Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression()

def load_model():
    # Load positive and negative tweets
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    # Creating labelled data
    documents = []

    # Adding positive tweets
    for tweet in positive_tweets:
        documents.append((tweet, "positive"))

    # Adding negative tweets
    for tweet in negative_tweets:
        documents.append((tweet, "negative"))

    # Split the dataset into the text and labels
    texts, labels = zip(*documents)

    # Split data into training and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.25, random_state=42)

    # Fit and transform the training data
    train_vectors = vectorizer.fit_transform(train_texts)

    # Transform the test data
    test_vectors = vectorizer.transform(test_texts)



    # Train the classifier
    logistic_classifier.fit(train_vectors, train_labels)


# Predict sentiments for test data using the trained classifier
def logistic_predict_sentiment(new_text):
    new_vector = vectorizer.transform([new_text])
    pred = logistic_classifier.predict(new_vector)
    return pred[0]


if __name__ == "__main__":
    load_model()
    # Test your results with the sample tweets below
    sample_tweets = [
        "Besides a soulful score by Ajay Atul, this adaptation of Ramayana is a baffling and collosal dissapointment.",
        "Had an amazing time at the beach today with friends. The weather was perfect! ☀️ #blessed",
        "Extremely disappointed with the service at the restaurant tonight. Waited over an hour and still got the order wrong. 😡",
        "Feeling really let down by the season finale. It was so rushed and left too many unanswered questions. 😞 #TVShow",
        "My phone keeps crashing after the latest update. So frustrating dealing with these glitches! 😠",
    ]

    # Test the function
    for sentence in sample_tweets:
        print(f"The sentiment predicted by the model is: {logistic_predict_sentiment(sentence)}")
        

# %%



