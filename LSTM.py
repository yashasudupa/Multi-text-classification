"""Supporting libraries."""
from flask import Flask, request, Response
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from tensorflow.keras.utils import get_file
import os
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
        RANDOM_STATE = 50
        LSTM_CELLS = 64
        VERBOSE = 0
        SAVE_MODEL = True
        EPOCHS = 6 # Early stopping criteria
        BATCH_SIZE = 2048

        train_csv = "train.csv"
        train_dataset = pd.read_csv(train_csv)
        train_labels = train_dataset['genres']

        # Encode labels of training data
        labels_en = data.encode_genres(train_labels)

        # Converts the training data from CSV format to dictionary
        train_dataset, train_labels, validation_dataset, validation_labels = data.csv_to_training_data(train_dataset, labels_en)

        # Tokenize the training data
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_tok)
        tokenizer.fit_on_texts(train_dataset)

        # Encode training data sequences into sequences
        train_sequences = tokenizer.texts_to_sequences(train_dataset)

        # Pad the training sequences
        train_padded = np.array(pad_sequences(train_sequences, maxlen=maxlen, padding=padding_type, truncating=trunc_type))

        # Validation sequence
        validation_sequences = tokenizer.texts_to_sequences(validation_dataset)
        validation_padded = np.array(pad_sequences(validation_sequences, maxlen=maxlen, padding=padding_type, truncating=trunc_type))

        # Train label sequence
        label_sequences = np.reshape(train_labels, (len(train_labels), )).astype(np.float32)
        validation_label_seq = np.reshape(validation_labels, (len(validation_labels), )).astype(np.float32)

        # Training the model
        model = tf.keras.Sequential([
        # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
        tf.keras.layers.Embedding(num_words, embedding_dim, input_length=len(data.common_genres)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        # use ReLU in place of tanh function since they are very good alternatives of each other.
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        # Add a Dense layer with 6 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        tf.keras.layers.Dense(len(data.common_genres), activation='softmax')
        ])
        model.summary()

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x=train_padded, y=label_sequences, epochs=EPOCHS, validation_data=(validation_padded, validation_label_seq), verbose=2)

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

        # Tokenize test data
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_tok)
        tokenizer.fit_on_texts(test_synopsis_list)
        test_sequences = tokenizer.texts_to_sequences(test_synopsis_list)

        # Pad the test sequence
        test_padded = np.array(pad_sequences(test_sequences, maxlen=maxlen, padding=padding_type, truncating=trunc_type))

        # Predict the test data
        y_test = model.predict_classes(x=test_padded, verbose=0)

        # Evaluate test data
        test_loss, test_acc = model.evaluate(x=test_padded, y=y_test, batch_size=None, verbose=1)

        print('Test Loss: {}'.format(test_loss))
        print('Test Accuracy: {}'.format(test_acc))

        # Trained model is used for predicting the test data
        y_pred = model.predict(x=test_padded, batch_size=None, verbose=1)

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
embedding_dim = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
maxlen = 188

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
