import numpy as np
import pandas as pd
import tensorflow as tf


class CSVDataset(object):

    # Determine CSV, label, and key columns
    CSV_COLUMNS = ["input",
                   "label"]
    LABEL_COLUMN = "label"

    # Set default values for each CSV column.
    # Treat is_male and plurality as strings.
    DEFAULTS = [["null"], [0]]

    @staticmethod
    def features_and_labels(row_data):
        label = row_data.pop(CSVDataset.LABEL_COLUMN)
        return row_data, label

    @staticmethod
    def input_fn(file: str, batch_size: int,  mode: str = 'eval'):
        dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=file,
            batch_size=batch_size,
            column_names=CSVDataset.CSV_COLUMNS,
            column_defaults=CSVDataset.DEFAULTS)

        dataset = dataset.map(map_func=CSVDataset.features_and_labels)

        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=batch_size).repeat()

        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset


class NumpyArrayDataset(object):

    @staticmethod
    def input_fn(X: np.ndarray, y: np.ndarray, batch_size: int,  mode: str = 'eval'):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=batch_size).repeat()

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset


class DataFrameDataset(object):

    LABEL_COLUMN = 'label'

    @staticmethod
    def features_and_labels(row_data):
        label = row_data.pop(DataFrameDataset.LABEL_COLUMN)
        return row_data, label

    @staticmethod
    def input_fn(df: pd.DataFrame, batch_size: int, mode: str = 'eval'):
        dataset = tf.data.Dataset.from_tensor_slices(dict(df))

        dataset = dataset.map(map_func=DataFrameDataset.features_and_labels)

        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=batch_size).repeat()

        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset



