Sentiment Analysis on IMDB Reviews

Project Overview:

This project builds a Sentiment Analysis Model using IMDB movie reviews to classify text as positive or negative. The model is trained using TF-IDF vectorization and a Logistic Regression classifier.

 📂 Project Steps

1️⃣ Data Collection

- We used the IMDB Reviews dataset from `tensorflow_datasets`.
- The dataset contains 50,000 labeled movie reviews (25,000 for training, 25,000 for testing).

 2️⃣ Text Preprocessing

To improve accuracy, we cleaned the text by:

-Tokenization: Splitting text into words.
- Stopword Removal: Removing common words like "the", "is", "and".
- Lemmatization: Converting words to their root form (e.g., "running" → "run").

3️⃣ Feature Engineering

We converted text into a numerical format using TF-IDF (Term Frequency - Inverse Document Frequency).

- Limited vocabulary to 5000 most important words.
- Converted reviews into a matrix of word importance scores.

 4️⃣ Model Training

- We trained a Logistic Regression classifier using the transformed text data.
- The model was fitted on the training set (25,000 reviews).

5️⃣ Model Evaluation

We evaluated the model using:

- Accuracy: `87.9%`
- Precision, Recall, F1-score:
- Negative Reviews → Precision: 0.88, Recall: 0.88, F1-score: 0.88
- Positive Reviews → Precision: 0.88, Recall: 0.88, F1-score: 0.88

6️⃣ Testing the Model

We tested the model on new, unseen text reviews. Example outputs:


Review: "Absolutely terrible. I regret wasting my time."
Predicted Sentiment: Negative

Review: "True to the book. Entertaining and fun."
Predicted Sentiment: Positive


🚀 How to Run the Project

This project was developed and tested in Google Colab.

Step 1: Running the Notebook

Open Google Colab or Jupyter Notebook


Step 2: Install Dependencies

If using Jupyter Notebook, install the required Python libraries:

pip install numpy pandas scikit-learn nltk tensorflow-datasets


step 3:Run All Cells

Simply run each cell in order, following the step-by-step workflow in the notebook.




📌 Observations

The model achieves 87.9% accuracy using TF-IDF and Logistic Regression.

The model successfully predicts positive and negative sentiments.

A few edge cases might be misclassified (e.g. sarcasm, neutral reviews).


Credits & References

Dataset: IMDB Reviews on TensorFlow Datasets

Libraries Used: NLTK, Scikit-learn, TensorFlow Datasets
