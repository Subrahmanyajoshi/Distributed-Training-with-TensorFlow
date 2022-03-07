from typing import List

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class HybridModel(object):

    def __init__(self, num_features: int, max_sequence_length: int):
        """ Init method
        Args:
            num_features (int): Total number of words
            max_sequence_length (int): Maximum allowed length for an inout sequence
        """
        self.num_features = num_features
        self.max_sequence_length = max_sequence_length

    def build(self, optimizer: str, loss: str, metrics: List, embedding_dim: int):
        """ Creates an hybrid (lstm + cnn) model, compiles it and returns it
        Args:
        Returns:
            Built model
        """
        print("[HybridModel::build] Building Hybrid model")
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(self.max_sequence_length,), name="input"))
        model.add(layers.Embedding(input_dim=self.num_features,
                                   output_dim=embedding_dim,
                                   input_length=self.max_sequence_length))
        model.add(layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.LSTM(128, recurrent_dropout=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
        return model
