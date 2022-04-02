import numpy as np
import pandas as pd
import tensorflow as tf


class CSVDataset(object):

    """ Create a tf dataset object from data in CSV files """

    # Determine CSV, label, and key columns
    CSV_COLUMNS = ["input",
                   "label"]
    LABEL_COLUMN = "label"

    # Set default values for each CSV column.
    # Treat is_male and plurality as strings.
    DEFAULTS = [["null"], [0]]

    @staticmethod
    def features_and_labels(row_data):
        """ Splits a single row into features and labels
        """
        label = row_data.pop(CSVDataset.LABEL_COLUMN)
        return row_data, label

    @staticmethod
    def input_fn(file: str, batch_size: int,  mode: str = 'eval'):
        """ Creates tf data object
        Args:
            file (str): Single file name or pattern
            batch_size (int): Train batch size
            mode (str): specified weather dataset to be created is train or validation dataset
        """
        dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=file,
            batch_size=batch_size,
            column_names=CSVDataset.CSV_COLUMNS,
            column_defaults=CSVDataset.DEFAULTS)

        dataset = dataset.map(map_func=CSVDataset.features_and_labels)

        # Shuffle and repeat infinitely if mode is train. Amount of data to be used per epoch will be controlled by
        # steps_per_epoch parameter during training
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=batch_size)
            dataset = dataset.repeat()

        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset


class NumpyArrayDataset(object):

    """ Create a tf dataset object from data in Numpy arrays """

    @staticmethod
    def input_fn(X: np.ndarray, y: np.ndarray, batch_size: int,  mode: str = 'eval'):
        """ Creates tf data object
        Args:
            X (np.ndarray): Array containing feature values
            y (np.ndarray): Array containing label values
            batch_size (int): Train batch size
            mode (str): specified weather dataset to be created is train or validation dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        # Shuffle and repeat infinitely if mode is train. Amount of data to be used per epoch will be controlled by
        # steps_per_epoch parameter during training
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=batch_size)
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset


class DataFrameDataset(object):

    """ Create a tf dataset object from data in data frames """

    LABEL_COLUMN = 'label'

    @staticmethod
    def features_and_labels(row_data):
        """ Splits a single row into features and labels
        """
        label = row_data.pop(DataFrameDataset.LABEL_COLUMN)
        return row_data, label

    @staticmethod
    def input_fn(df: pd.DataFrame, batch_size: int, mode: str = 'eval'):
        """ Creates tf data object
        Args:
            df (np.ndarray): Dataframe containing train/validation data
            batch_size (int): Train batch size
            mode (str): specified weather dataset to be created is train or validation dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices(dict(df))
        dataset = dataset.map(map_func=DataFrameDataset.features_and_labels)

        # Shuffle and repeat infinitely if mode is train. Amount of data to be used per epoch will be controlled by
        # steps_per_epoch parameter during training
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=batch_size)
            dataset = dataset.repeat()

        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset



