# Movie Review Sentiment Analysis

## Overview

This project focuses on sentiment analysis of movie reviews extracted from IMDb. The sentiment analysis is performed using a combination of web scraping, natural language processing (NLP) techniques, and machine learning algorithms. The goal is to determine the sentiment (positive or negative) of each movie review.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Project Structure](#project-structure)
- [Web Scraping](#web-scraping)
- [Data Preprocessing](#data-preprocessing)
- [Sentiment Analysis](#sentiment-analysis)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Screenshots](#screenshots)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook (for running the project interactively)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Rahul-Sahani04/MovieReviewSentimentAnalysis.git
    ```
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

4. Open and Run all the cell in the MovieReview.py


## Project Structure

The project is organized into several components to facilitate a structured and modular approach.

- **`web_scraping.ipynb`**: Jupyter Notebook dedicated to web scraping IMDb movie reviews.
- **`data_preprocessing.ipynb`**: Jupyter Notebook focused on cleaning and preparing the scraped data through text cleaning, tokenization, and lemmatization.
- **`sentiment_analysis.ipynb`**: Jupyter Notebook handling sentiment analysis using the VADER sentiment analysis tool, with a focus on classifying reviews as positive or negative.
- **`data/`**: Directory containing scraped and processed data for reference.
- **`images/`**: Directory storing visualizations or images generated during the analysis.
- **`README.md`**: Project documentation providing an overview and guidance.

### Web Scraping

The `web_scraping.ipynb` notebook utilizes the BeautifulSoup library to scrape IMDb movie reviews. Reviews are extracted from the IMDb website and stored for subsequent analysis.

### Data Preprocessing

The `data_preprocessing.ipynb` notebook concentrates on the cleaning and preparation of the scraped data. Techniques such as text cleaning, tokenization, and lemmatization are applied to enhance the quality of the textual data.

### Sentiment Analysis

In the `sentiment_analysis.ipynb` notebook, sentiment analysis is conducted using the VADER sentiment analysis tool. The compound score is employed to classify each review as positive or negative, providing a comprehensive understanding of the sentiments expressed.

### Model Evaluation

The performance of the sentiment analysis model is thoroughly evaluated in terms of accuracy and classification reports. These results offer insights into the model's effectiveness in predicting sentiment.

### Results

Visualizations and results of the sentiment analysis are stored in the `results/` directory. These include the distribution of predicted sentiments and other relevant visualizations that aid in interpreting the sentiment analysis outcomes.

### Screenshots

Step 1:
![Screenshot 1](https://github.com/Rahul-Sahani04/MovieReviewSentimentAnalysis/blob/main/screenshots/Streamlit1.png?raw=true)


Step 2:
![Screenshot 2](https://github.com/Rahul-Sahani04/MovieReviewSentimentAnalysis/blob/main/screenshots/Streamlit2.png?raw=true)

### How to Use

To utilize the project, follow the steps outlined in the [Getting Started](#getting-started) section to set up the environment locally. Execute the Jupyter Notebooks in the specified order (`web_scraping.ipynb`, `data_preprocessing.ipynb`, `sentiment_analysis.ipynb`) to perform sentiment analysis on IMDb movie reviews.

### Contributing

If you wish to contribute to the project, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines for details on how to get involved and contribute effectively.

### License

This project is licensed under the MIT License, ensuring an open and collaborative development environment.
