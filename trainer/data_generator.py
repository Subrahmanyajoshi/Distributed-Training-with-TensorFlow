from typing import Dict

import numpy as np
from google.cloud.storage import Bucket
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):

    def __init__(self, input_text: np.ndarray, labels: np.ndarray, batch_size: int):
        """ Init Method
        Args:
            input_text (np.array): numpy array of input texts
            labels (np.array): labels associated with filenames
            batch_size (int): batch size of model
            bucket (Bucket): Gcs bucket name
        """
        self.input = input_text
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.input) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx: int):
        """ Creates a batch of images and associated labels and returns it
        Args:
            idx (int): index from which to start the batch
        Returns:
            A batch of tokenized text and labels associated with it
        """
        batch_x = self.input[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = [[i] for i in batch_y]
        return np.array(batch_x), np.array(batch_y)
