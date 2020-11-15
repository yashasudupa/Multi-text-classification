"""Supporting libraries."""
import re
from flask import Flask, request, Response
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
import os
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from data_handling import Data

app = Flask(__name__)

# ----------------------------------------------------
# demonstrates to build a Machine Learning model
# to predict the genres of a movie, given its synopsis
#
# Author : Yashas Nagaraj Udupa
# ----------------------------------------------------


@app.route('/genres/train', methods=['GET', 'POST'])
def train_model():
    """
    CSV with header 'movie_id', 'synopsis' and 'genres'.

    FLASK API that POST CSV to a training endpoint
    at localhost:5000/genres/train.
    """
    if request.method == 'POST':
        global model, num_words, embedding_dim, trunc_type, padding_type, oov_tok, maxlen
        
        EPOCHS = 6 # Early stopping criteria

        vectorizer = TfidVectorizer()

        train_csv = "train.csv"
        train_dataset = pd.read_csv(train_csv)
        train_labels = train_dataset['genres']

        # Encode labels of training data
        labels_en = data.encode_genres(train_labels)

        # Converts the training data from CSV format to dictionary
        train_dataset, train_labels, validation_dataset, validation_labels = data.csv_to_training_data(train_dataset, labels_en)

        # Term-Document matrix is created
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(train_dataset).toarray()
        df_new = pd.DataFrame(X, columns=vectorizer.get_feature_names())

        # Model definition 
        X = df_new
        y = train_labels

        kfold = kFold(n_splits=10)

        nb_multinomial = MultinomialNB()
        #nb_bernoulli = BernoulliNB()

        # As a storage of the model's performance
        metrics = []

        for train_idx in X:
            X_train = X[train_idx]
            y_train = y[train_idx]
            nb_multinomial.fit(X_train, y_train)   

        # Retrieve the mean of the result
        print("%.3f" % np.array(metrics).mean())

        # Training response is returned to the server
        return Response(
            status=200,
            headers={
                "message": "The model is successfully trained"
            })


@app.route('/genres/predict', methods=['GET', 'POST'])
def predict_model():
    """
    FLASK API that POST a CSV with header 'movie_id' and 'synopsis'.

    Returns a CSV with header movie_id and predicted_genres
    at localhost:5000/genres/train.
    """
    if request.method == 'POST':
        global data
        test_csv_path = "test.csv"

        # Converts the test data from CSV format to dictionary
        test_dataset = pd.read_csv(test_csv_path)

        # Filter the test synopsis from stopping words
        test_synopsis_list = data.filter_synopsis(test_dataset)

        # Term-Document matrix is created
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(test_dataset).toarray()
        df_new = pd.DataFrame(X, columns=vectorizer.get_feature_names())

        X = df_new
        kfold = KFold(n_splits=10)
        
        # Model definition
        nb_multinomial = MultinomialNB()
        #nb_bernoulli = BernoulliNB()
        
        # As a storage of the model's performance        
        for test_idx in kfold.split(X):
            X_test = X[test_idx], X[test_idx]
            y_pred = model.predict(X_test)       

        # Probability distribution is converted to it's appropriate test class
        # and top 5 predicted classes are retrieved
        predicted_genres = data.get_unencoded_genres(y_pred)

        # Predicted genres are embedded with the respective movie id
        # in test dataset and stored in submission variable
        test_dataset['predicted_genres'] = predicted_genres
        submission = test_dataset[['movie_id', 'predicted_genres']]

        # Data in 'submission' varaible is stored in submission.csv
        submission.to_csv("submission.csv", index=False)

        # Prediction response is returned to the server
        return Response(
            submission.to_csv(index=False),
            status=200,
            headers={
                "description": "The top 5 predicted movie genres",
                "Content-Type": "text/csv",
            })


model = None
data = Data()
num_words = 5000

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False)
