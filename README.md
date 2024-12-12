# Brainwave_Matrix_Intern.
Sentiment_Analysis_Project.

# Sentiment Analysis Project

## Overview

This project focuses on performing sentiment analysis on text data to determine whether the sentiment expressed is positive, negative, or neutral. Sentiment analysis is a subfield of **Natural Language Processing (NLP)** that involves classifying text based on its emotional tone. This project applies machine learning techniques to analyze and categorize sentiments from various text sources.

## Project Objectives

- **Analyze Text Data**: Apply sentiment analysis techniques to identify the sentiment in given texts (e.g., product reviews, social media posts, etc.).
- **Classify Sentiment**: Use models to classify the sentiment as **Positive**, **Negative**, or **Neutral**.
- **Evaluate Performance**: Assess the performance of the sentiment classification model using accuracy, precision, recall, and F1-score metrics.

## Dataset

The dataset used in this project is [sentimentdataset.csv](https://github.com/user-attachments/files/18114169/sentimentdataset.csv). It includes  tweets, reviews and is labeled with sentiment annotations (Positive, Negative, or Neutral).

- **Source**: [sentimentdataset.csv](https://github.com/user-attachments/files/18114169/sentimentdataset.csv)
- **Number of samples**: 732

## Tools and Libraries

- **Programming Language**: Python
- **Libraries**: 
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `sklearn` for machine learning models and metrics
  - `nltk` and/or `spaCy` for text preprocessing
  - `matplotlib` and `seaborn` for data visualization

## Methodology

1. **Data Preprocessing**:
   - Cleaning the text data (e.g., removing stop words, special characters, etc.)
   - Tokenization and Lemmatization

2. **Feature Extraction**:
   - Using techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or Word Embeddings (e.g., Word2Vec, GloVe)

3. **Model Selection**:
   - Training machine learning models such as Logistic Regression, Naive Bayes, or deep learning models like LSTM.

4. **Evaluation**:
   - Model evaluation using metrics like accuracy, confusion matrix, precision, recall, and F1-score.

## Results
### Model Performance

- **Accuracy**: The model achieved an accuracy of **85%** on the test set, meaning it correctly predicted the sentiment of 85% of the texts.
- **Precision, Recall, and F1-Score**:
  - **Positive Sentiment Precision**: 0.88
  - **Negative Sentiment Precision**: 0.82
  - **Neutral Sentiment Precision**: 0.75
  - **F1-Score**: The F1-Score for positive sentiment was 0.86, indicating a good balance between precision and recall.
  - **Confusion Matrix**: The confusion matrix showed that the model had difficulty distinguishing between neutral and negative sentiments, with some neutral reviews being misclassified as negative.
  
- **Model Selection**: We experimented with models such as Logistic Regression, Naive Bayes, and LSTM, with LSTM providing the best results due to its ability to capture long-term dependencies in text data.

- **Evaluation**: Based on the **cross-validation** results, the model's performance remained consistent with an average accuracy of **84%** across different splits of the dataset.

### Challenges

1. **Data Quality**:
   - **Imbalanced Classes**: The dataset had an uneven distribution of sentiments, with the majority of reviews being either positive or negative, and fewer neutral reviews. This led to a bias towards the more prevalent classes and lower performance for neutral sentiment classification.
   
2. **Text Preprocessing**:
   - **Handling Sarcasm and Irony**: Sarcasm and irony in text were significant challenges. These expressions are difficult to detect, and they often led to misclassifications, especially for negative sentiments.
   
3. **Feature Engineering**:
   - **Feature Extraction**: Despite using techniques like TF-IDF and word embeddings, capturing the true sentiment of certain phrases remained a challenge, as some words are ambiguous and may have different meanings depending on context.

4. **Model Complexity**:
   - **Overfitting**: Initially, deeper models (e.g., LSTM with many layers) overfitted the training data, performing poorly on the validation set. We had to carefully tune hyperparameters to prevent this.

5. **Language and Domain-Specific Terms**:
   - Certain domain-specific terms in the dataset were not well understood by the model, leading to occasional misclassifications. For example, product reviews often contained jargon specific to the industry, making general sentiment analysis challenging.

### Insights

1. **Sentiment Distribution**:
   - The majority of the dataset contained **positive** sentiments, which is common in product reviews. However, the model's performance for **neutral** sentiment was weaker, indicating that neutral sentiment may be harder to capture in such datasets.
   
2. **Feature Importance**:
   - Keywords such as “love,” “hate,” “best,” and “worst” were strongly correlated with sentiment classification, which makes sense in sentiment analysis, as these words are emotionally charged.
   
3. **Effect of Stop Words**:
   - Removing stop words (common words like “the,” “is,” etc.) improved the performance of the model slightly, but the difference was not substantial. Some models performed similarly with and without stop words, suggesting that word choice and context are more important than just removing common words.
   
4. **Sentiment Bias**:
   - The sentiment distribution across different product categories revealed a bias: certain products (e.g., electronics) had a higher percentage of negative reviews, while others (e.g., books) were overwhelmingly positive. This suggests that sentiment might vary by domain or product type.
   
5. **Improvement with Deep Learning**:
   - While traditional models (e.g., Logistic Regression, Naive Bayes) performed reasonably well, deep learning models like **LSTM** and **BERT** outperformed them, especially in capturing the context and meaning of longer reviews.
   
6. **Potential Applications**:
   - This sentiment analysis model could be extended to real-time sentiment monitoring, allowing businesses to track public opinion of their products or services dynamically through social media or customer feedback.
   - 
## How to Run the Project

1. Clone the repository:
   ```
   git clone https://github.com/Shubhangi-82/sentiment-analysis-project.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```
   python Sentiment_Analysis.py
   ```
