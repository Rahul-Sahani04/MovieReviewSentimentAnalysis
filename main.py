import os
import matplotlib.pyplot as plt
from MovieReview import scrape_imdb_reviews
from TwitterBasedMovieReview import load_model, logistic_predict_sentiment
import pandas as pd

def main():
    # IMDb movie URL (replace with the actual movie URL)
    imdb_code = "tt4633694"
    movie_url = f'https://www.imdb.com/title/{imdb_code}/reviews'

    # Scrape IMDb movie reviews
    movie_reviews = scrape_imdb_reviews(movie_url)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(movie_reviews)

    # Load sentiment analysis model
    load_model()

    # Predict sentiment for each review
    df["sentiment"] = df["text"].apply(logistic_predict_sentiment)

    # Save DataFrame to CSV file
    save_path = "output"
    os.makedirs(save_path, exist_ok=True)  # Create the 'output' directory if it doesn't exist
    df.to_csv(os.path.join(save_path, "movie_reviews_with_sentiment.csv"), index=False)

    # Visualize sentiment distribution
    visualize_sentiment_distribution(df["sentiment"], save_path)

def visualize_sentiment_distribution(sentiments, save_path):
    # Determine colors dynamically based on column names
    colors = ["green" if col.lower() == "positive" else "red" for col in sentiments.value_counts().index]

    plt.figure(figsize=(8, 6))
    sentiments.value_counts().plot(kind="bar", color=colors, alpha=0.7)
    plt.title("Sentiment Distribution of Movie Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(os.path.join(save_path, "sentiment_distribution.png"))
    plt.show()

if __name__ == "__main__":
    main()
