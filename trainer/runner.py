import argparse
import os
import zipfile
from argparse import Namespace
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from tf_datasets import NumpyArrayDataset
from trainer.gcs_callback import GCSCallback
from trainer.models import HybridModel


class TokenizerDetails(object):

    def __init__(self, **kwargs):
        self.tokenizer = kwargs.get('tokenizer', None)
        self.top_k = kwargs.get('top_k', 20000)
        self.max_sequence_length = kwargs.get('max_sequence_length', 500)


class Trainer(object):
    model_NAME = 'Amazon_Reviews_Analysis.hdf5'
    TOP_K = 20000
    MAX_SEQUENCE_LENGTH = 500

    def __init__(self, arguments: Namespace):
        self.tokenizer = Tokenizer(num_words=Trainer.TOP_K)
        self.data_dir = arguments.data_dir
        self.batch_size = arguments.batch_size
        self.output_dir = os.path.join(arguments.save_dir, f"{datetime.now().strftime('%Y_%m_%d-%H:%M:%S')}")

    @staticmethod
    def clean_up():
        """ Deletes temporary directories created while training"""

        print(f"[Trainer::cleanup] Cleaning up...")
        os.system('rm *.csv.gz')
        os.system('rm -rf trained_model')

    def load_data(self):
        print(f"[Trainer::load_data] Copying data from {self.data_dir} to here...")
        os.system(f"gsutil -m cp -r "
                  f"{os.path.join(self.data_dir, 'train_val.zip')} ./")
        with zipfile.ZipFile('train_val.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
        os.system('rm -f train_val.zip')

    def preprocess(self) -> Tuple:
        train_df = pd.read_csv('train_text.csv.gz')
        val_df = pd.read_csv('val_text.csv.gz')
        lines = list(train_df['input']) + list(val_df['input'])

        print("[Trainer::preprocess] Fitting tokenizer on texts...")
        self.tokenizer.fit_on_texts(lines)
        print(f"[Trainer::preprocess] Size of word index: {len(self.tokenizer.word_index)}")

        print("[Trainer::preprocess] Converting texts to sequences...")
        X_train = self.tokenizer.texts_to_sequences(list(train_df['input']))
        X_val = self.tokenizer.texts_to_sequences(list(val_df['input']))

        print("[Trainer::preprocess] Padding sequences so that they have the same length...")
        X_train = sequence.pad_sequences(X_train, maxlen=Trainer.MAX_SEQUENCE_LENGTH)
        X_val = sequence.pad_sequences(X_val, maxlen=Trainer.MAX_SEQUENCE_LENGTH)

        y_train = np.array(train_df['labels'])
        y_val = np.array(val_df['labels'])

        with open(os.path.join('parser_output', 'word_index.txt'), 'w') as fstream:
            for word, index in self.tokenizer.word_index.items():
                if index < Trainer.TOP_K:  # only save mappings for TOP_K words
                    fstream.write("{}:{}\n".format(word, index))
        print("[Trainer::preprocess] Dumped word index to word_index.txt")

        return X_train, y_train, X_val, y_val

    def train(self):

        print("[Trainer::train] Loaded data")
        X_train, y_train, X_val, y_val = self.preprocess()

        train_dataset = NumpyArrayDataset.input_fn(X=X_train, y=y_train, batch_size=self.batch_size, mode='train')
        val_dataset = NumpyArrayDataset.input_fn(X=X_val, y=y_val, batch_size=self.batch_size, mode='eval')

        num_features = min(len(self.tokenizer.word_index) + 1, Trainer.TOP_K)

        model = HybridModel(num_features=num_features,
                            max_sequence_length=Trainer.MAX_SEQUENCE_LENGTH).build(optimizer='adam',
                                                                                   loss="binary_crossentropy",
                                                                                   metrics=["accuracy"],
                                                                                   embedding_dim=200)
        model.summary()
        print(f"[Trainer::train] Built Hybrid model")

        cp_callback = ModelCheckpoint(filepath='/tmp/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_accuracy',
                                      save_freq='epoch', verbose=1, period=1,
                                      save_best_only=False, save_weights_only=True)
        gcs_callback = GCSCallback(cp_path='gs://text-analysis-323506/checkpoints', bucket_name='text-analysis-323506')

        print("[Trainer::train] Started training")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            callbacks=[cp_callback, gcs_callback]
        )

        # save model as hdf5 file
        os.makedirs('trained_model')
        os.makedirs(os.path.join('trained_model', datetime.now().strftime("%Y_%m_%d-%H:%M:%S")))
        model_path = os.path.join('trained_model', f"Hybrid_{Trainer.model_NAME}")
        model.save_weights(model_path)

        print(f"[Trainer::train] Copying trained model to {self.output_dir}")
        os.system('gsutil -m mv trained_model {self.output_dir}')

        print("Saving model in TF saved model format")
        model.save(os.path.join(self.output_dir, f"{datetime.now().strftime('%Y_%m_%d-%H:%M:%S')}", "saved_model"))


def main():
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument('--package-path', help='GCS or local path to training data',
                        required=False)
    parser.add_argument('--job-dir', type=str, help='GCS location to write checkpoints and export models',
                        required=False)
    parser.add_argument('--data-dir', type=str, help='GCS location, where data is stored',
                        required=True)
    parser.add_argument('--save-dir', type=str, help='GCS location to store trained model',
                        required=True)
    parser.add_argument('--batch-size', type=str, help='batch size',
                        required=True)

    args = parser.parse_args()

    trainer = Trainer(arguments=args)
    trainer.train()


if __name__ == "__main__":
    main()
