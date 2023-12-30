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

nltk.download("wordnet")
nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download("stopwords")

movie_name = ""


# Function to scrape IMDb movie reviews
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup
import requests

def scrape_imdb_reviews(movie_url, no_of_pages):
    global movie_name
    global my_bar
    # Set up ChromeOptions for headless browsing
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")  # Necessary for headless mode on some systems
    chrome_options.add_argument("--window-size=1920x1080")  # Set a reasonable window size

    # Create a headless WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Navigate to the movie URL using the headless WebDriver
        driver.get(movie_url)

        # Extracting movie name
        movie_element = driver.find_element(By.CSS_SELECTOR, "h3[itemprop='name'] a")
        movie_name = movie_element.text.strip()

        # Print the movie name (optional)
        print(f"Movie Name: {movie_name}")
        
        NoOfLoadMore = 0
        per_cycle_increment = 100 / int(no_of_pages)
        
        # Click "load more" button until the user wants
        while NoOfLoadMore < int(no_of_pages):
            try:
                my_bar.progress(int(per_cycle_increment) * NoOfLoadMore, text=progress_text)
                # Extracting "load more" button and clicking it
                loadMoreElement = driver.find_element(By.ID, "load-more-trigger")
                loadMoreElement.click()
                NoOfLoadMore += 1

                # Wait for dynamic content to load (adjust as needed)
                driver.implicitly_wait(5)

            except NoSuchElementException:
                # Break the loop when the "load more" button is not found
                break
            

        # Extracting review text after all "load more" clicks
        soup = BeautifulSoup(driver.page_source, "html.parser")
        reviews = [
            {"text": review_div.get_text(strip=True)}
            for review_div in soup.find_all("a", class_="title")
        ]
        
        my_bar.empty()

        return reviews

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the WebDriver after scraping
        driver.quit()


# Function for text cleaning
def clean_text(text):
    cleaned_text = re.sub(
        r"[^\w\s]", "", text.lower()
    )  # Convert to lowercase and remove special characters
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
def perform_sentiment_analysis(movie_code, num_of_pages):
    # IMDb movie URL
    movie_url = f"https://www.imdb.com/title/{movie_code}/reviews"

    # Scrape IMDb movie reviews
    movie_reviews = scrape_imdb_reviews(movie_url, num_of_pages)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(movie_reviews)

    # Data Preprocessing
    df["text_cleaned"] = df["text"].apply(clean_text)
    df["tokens"] = df["text_cleaned"].apply(tokenize_text)
    df["tokens"] = df["tokens"].apply(remove_stopwords)
    df["tokens"] = df["tokens"].apply(lemmatize_text)

    # Sentiment Analysis using VADER
    vader = SentimentIntensityAnalyzer()
    df["compound"] = df["text_cleaned"].apply(
        lambda x: vader.polarity_scores(x)["compound"]
    )

    # Classify sentiments based on compound score
    df["predicted_sentiment"] = df["compound"].apply(
        lambda x: "positive" if x >= 0 else "negative"
    )

    # Save results to CSV
    save_path = "results"
    df.to_csv(
        os.path.join(save_path, f"movie_reviews_with_sentiment_for_{movie_code}.csv"),
        index=False,
    )

    # Visualize the distribution of predicted sentiments
    colors = [
        "green" if col.lower() == "positive" else "red"
        for col in df["predicted_sentiment"].value_counts().index
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    df["predicted_sentiment"].value_counts().plot(kind="bar", color=colors, ax=ax)
    ax.set_title(f"Distribution of Predicted Sentiments in IMDb Reviews ({movie_code})")
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("Count")
    plt.tight_layout()

    # Save the plot to a file
    plot_file_path = os.path.join(
        save_path, f"sentiment_distribution_for_{movie_code}.png"
    )
    plt.savefig(plot_file_path)

    # Display a message indicating positive sentiment
    more_counts = df["predicted_sentiment"].value_counts().index
    positive_message = (
        "The movie has positive reviews! You should consider watching it."
    )
    negative_message = "The sentiment analysis did not identify a clear positive sentiment in the reviews."

    st.title("IMDb Movie Reviews Sentiment Analysis")
    st.subheader(f"Movie Name: {movie_name}")
    st.image(
        plot_file_path,
        caption=f"Distribution of Predicted Sentiments in IMDb Reviews ({movie_code})",
        use_column_width=True,
    )
    st.subheader("Sentiment Analysis Results:")
    st.write(df[["text", "predicted_sentiment"]])
    st.subheader("Sentiment Overview:")
    if "positive" == more_counts[0]:
        st.success(positive_message)
    else:
        st.warning(negative_message)

    st.write(f"You can view the reviews [here]({movie_url})")


# Streamlit app
st.title("IMDb Movie Reviews Sentiment Analysis")

GetCode = st.image(
    "screenshots/HowToGetIMDBCode.png",
    caption="How To Get IMDB Code",
    use_column_width=True,
)

# Input field for IMDb code
movie_code = st.text_input("Enter IMDb Code (e.g., tt12915716, tt0111161):")
num_of_pages = st.text_input("Enter No. of Review Pages to scrape (Enter 0 for scraping all the reviews <- Takes more time):")

progress_text = "Operation in progress. Please wait."

# Button to trigger sentiment analysis
if st.button("Perform Sentiment Analysis"):
    if movie_code:
        GetCode.empty()
        my_bar = st.progress(0, text=progress_text)
        # with st.spinner('Loading...'):
        perform_sentiment_analysis(movie_code, num_of_pages)
    else:
        st.warning("Please enter a valid IMDb Code.")
