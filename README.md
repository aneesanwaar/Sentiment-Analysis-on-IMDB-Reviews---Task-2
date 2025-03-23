# Sentiment Analysis on IMDB Reviews

## Project Overview:

This project builds a Sentiment Analysis Model using IMDB movie reviews to classify text as positive or negative. The model is trained using TF-IDF vectorization and a Logistic Regression classifier.

## Project Steps

### 1. Data Collection

- I used the IMDB Reviews dataset from `tensorflow_datasets`.
- The dataset contains 50,000 labeled movie reviews (25,000 for training, 25,000 for testing).

### 2. Text Preprocessing

To improve accuracy, I cleaned the text by:

-Tokenization: Splitting text into words.
- Stopword Removal: Removing common words like "the", "is", "and".
- Lemmatization: Converting words to their root form (e.g., "running" → "run").

### 3. Feature Engineering

I converted text into a numerical format using TF-IDF (Term Frequency - Inverse Document Frequency).

- Limited vocabulary to 5000 most important words.
- Converted reviews into a matrix of word importance scores.

### 4. Model Training

- I trained a Logistic Regression classifier using the transformed text data.
- The model was fitted on the training set (25,000 reviews).

### 5. Model Evaluation

I evaluated the model using:

- Accuracy: `87.9%`
- Precision, Recall, F1-score:

Sentiment	Precision	Recall	F1-score
Negative	0.88	0.88	0.88
Positive	0.88	0.88	0.88

### 6. Testing the Model

I tested the model on new, unseen text reviews. Example outputs:


Review: "Absolutely terrible. I regret wasting my time."
Predicted Sentiment: Negative

Review: "True to the book. Entertaining and fun."
Predicted Sentiment: Positive


## How to Run the Project


Step 1: Open the Notebook

Open Google Colab and upload the .ipynb file.

Or, run locally in Jupyter Notebook using:

```
jupyter notebook
```
Step 2: Install Dependencies

```
pip install numpy pandas scikit-learn nltk tensorflow-datasets
```

step 3: Run All Cells

Simply run each cell in order, following the step-by-step workflow in the notebook.


## Observations:

1.The model achieves 87.9% accuracy using TF-IDF and Logistic Regression.
2.The model successfully predicts positive and negative sentiments.
3.A few edge cases might be misclassified (e.g. sarcasm, neutral reviews).



## Credits and References:

Dataset: IMDB Reviews on TensorFlow Datasets

Libraries Used: NLTK, Scikit-learn, TensorFlow Datasets.
