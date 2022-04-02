from tensorflow.keras.callbacks import Callback
from google.cloud import storage
import os


class GCSCallback(Callback):
    """ A custom callback to copy checkpoints from local file system directory to Google Cloud Storage directory"""

    def __init__(self, cp_path: str, bucket_name: str):
        """ init method
        Args:
            cp_path (str): gcs directory path to store checkpoints
            bucket_name (str): name of GCS bucket
        """
        super(GCSCallback, self).__init__()
        self.checkpoint_path = cp_path
        self.bucket_name = bucket_name

        client = storage.Client()
        self.bucket = client.get_bucket(bucket_name)

    def upload_file_to_gcs(self, src_path: str, dest_path: str):
        """ Uploads file to Google Cloud Storage
        Args:
            src_path (str): absolute path of source file
            dest_path (str): gcs directory path beginning with 'gs://<bucket-name>'
        Returns:
        """
        # blob needs only the path inside the bucket. we need to remove gs://<bucket-name> part
        dest_path = dest_path.split(f'{self.bucket_name}/')[1]

        # Create a complete destination path. This is basically self.cp_path + file_name.
        dest_path = os.path.join(dest_path, os.path.basename(src_path))

        blob = self.bucket.blob(dest_path)
        blob.upload_from_filename(src_path)

    def on_epoch_end(self, epoch, logs=None):
        """ Copies checkpoints created in /tmp/checkpoints directory to destination GCS directory"""

        # ModelCheckpoint callback will write checkpoints to /tmp/checkpoints directory
        for cp_file in os.listdir('/tmp/checkpoints'):
            src_path = os.path.join('/tmp/checkpoints', cp_file)
            self.upload_file_to_gcs(src_path=src_path, dest_path=self.checkpoint_path)
