# 🎬 RNN for IMDB Sentiment Analysis

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A PyTorch-based Recurrent Neural Network (RNN) designed to classify the sentiment of movie reviews from the IMDB dataset as either **positive** or **negative**. This project demonstrates the end-to-end pipeline of NLP preprocessing, text vectorization (TF-IDF), dataset curation using PyTorch `TensorDataset`, and training a custom Recurrent Neural Network.

---

## 📊 Dataset Overview

The project uses the **IMDB Dataset of 50K Movie Reviews** for sentiment analysis.

- **Features**: `review` (textual review of the movie)
- **Target**: `sentiment` (labeled as `positive` or `negative`)
- **Size**: ~50,000 highly polar movie reviews for training and testing.

> ⚠️ **Note**: The dataset (`IMDB Dataset.csv`) is excluded from this repository via `.gitignore` due to file size constraints.

---

## 🛠️ Natural Language Processing Pipeline

Before feeding text into the RNN model, the reviews go through a rigorous preprocessing pipeline:

1. **Lowercasing**: Standardizing all review text to lowercase.
2. **Cleaning**: Removing HTML tags, URLs (http/https), and punctuation marks using Regex.
3. **Stopword Removal**: Eliminating non-informative words using the NLTK library.
4. **Stemming**: Reducing words to their base form using the `PorterStemmer`.
5. **Encoding**: Encoding categorical labels (`positive` -> `1`, `negative` -> `0`).
6. **Vectorization**: Transforming the text corpus into numerical vectors using a TF-IDF Vectorizer (top 5,000 features).

---

## 🧠 Model Architecture

The custom RNN classifier is implemented in PyTorch and structured as follows:

```mermaid
graph TD
    A[Input Features: 5000 TF-IDF] --> B[RNN Layer: 128 Hidden Units]
    B --> C[Fully Connected Layer]
    C --> D[Sigmoid Activation]
    D --> E[Output Sentiment: 0 or 1]
```

## 🧠 Model Architecture

![RNN Architecture](rnn_architecture.png)

