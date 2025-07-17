# Twitter Sentiment Analysis

A sentiment analysis project that classifies tweets as positive or negative using machine learning and NLP techniques, deployed as an interactive web application.

## Project Overview

This project provides a robust solution for analyzing the sentiment of textual data, specifically Twitter tweets. It showcases the entire machine learning pipeline, from raw data to a user-friendly deployment. The core of the project involves building and deploying a sentiment classification model capable of distinguishing between positive and negative sentiments in real-time.

## Key Features

* **Sentiment Classification:** Accurately classifies tweet sentiment as positive or negative.
* **Data Preprocessing:** Implements advanced NLP techniques for cleaning and preparing text data.
* **Machine Learning Model:** Utilizes a Logistic Regression model for efficient sentiment prediction.
* **Interactive Web Application:** Deploys the model as a user-friendly web interface using Streamlit.

## Setup and Installation

Due to GitHub's file size limit, the dataset (`training.1600000.processed.noemoticon.csv`) is not included in this repository.
1.  **Download the dataset:** Obtain it from [Kaggle - Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) and place it in the `data/` folder (create this folder if it doesn't exist).
2.  **Train the Model:** Run the Jupyter Notebook (`model.ipynb`) to preprocess the data and train the model. This will generate `model.pkl` and `vectorizer.pkl`.

## Running the App

To start the Streamlit app:
```bash
python3 -m streamlit run app.py
