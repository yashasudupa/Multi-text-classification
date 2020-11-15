import pandas as pd
import numpy as np
from sklearn.feature_extraction import stop_words
from sklearn import preprocessing

class Data:
    """Data handler."""

    def __init__(self):
        """Initialization."""
        self.training_portion = .8
        self.common_genres = ['Drama', 'Action', 'Crime', 'Horror', 'Comedy', 'Animation',
                             'Romance', 'Thriller', 'Documentary', 'Adventure', 'Fantasy',
                             'Western', 'Sci-Fi', 'Musical', 'Mystery', 'Children', 'War',
                             'IMAX', 'Film-Noir']
        self.le = preprocessing.LabelEncoder()
        
    def encode_genres(self, train_labels):
        """To encode the genres."""
        for index, label in enumerate(train_labels):
            train_labels[index] = self.common_genres[self.common_genres.index(label.split()[0])]
        self.le.fit(train_labels)
        labels_en = self.le.transform(train_labels)
        return labels_en

    def filter_synopsis(self, dataset):
        """To filter synopsis from stopping words."""
        dataset_synopsis_list = []
        for index, row in enumerate(dataset['synopsis']):
            for word in stop_words.ENGLISH_STOP_WORDS:
                token = ' ' + word + ' '
                dataset['synopsis'][index] = row.replace(token, ' ')
                # train_dataset['synopsis'][index] = row.replace(' ',' ')
            dataset_synopsis_list.append(dataset['synopsis'][index])
        return dataset_synopsis_list

    def csv_to_training_data(self, train_dataset, labels_en):
        """To convert the data frame from CSV format to dictionary."""
        train_synopsis_list = self.filter_synopsis(train_dataset)
        train_size = int(len(train_dataset['synopsis']) * self.training_portion)
        train_dataset = train_synopsis_list[0: train_size]
        train_labels = labels_en[0: train_size]
        validation_dataset = train_synopsis_list[train_size:]
        validation_labels = labels_en[train_size:]
        return train_dataset, train_labels, validation_dataset, validation_labels

    def get_unencoded_genres(self, predicted_genres):
        """To unencode the genres."""
        genres = np.asarray(self.le.classes_)
        final_prediction = []
        for predicted_row in predicted_genres:
            ind = np.argpartition(predicted_row, -4)[-5:]
            ind = np.flip(ind)
            prediction_text = " ".join(genres[ind])
            final_prediction.append(prediction_text)

        return np.asarray(final_prediction)


