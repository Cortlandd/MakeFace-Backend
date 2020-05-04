from app import app
from google.cloud import storage
import os
import imageio
imageio.plugins.ffmpeg.download()

MODEL_NAME = 'vox-cpk.pth.tar'

def download_model():

    IMPORTANT_FOLDER = 'important'
    CLOUD_STORAGE_BUCKET = 'heroic-gantry-275322.appspot.com'

    # Create a Cloud Storage client.
    gcs = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)

    model_blob = bucket.blob(IMPORTANT_FOLDER + '/' + MODEL_NAME)
    model_blob.download_to_filename('app/ml/' + MODEL_NAME, raw_download=True)

    return

if __name__ == '__main__':

    # download model weights if not already downloaded
    model_found = 0
    files = os.listdir("app/ml")
    
    if MODEL_NAME in files:
        model_found = 1

    if model_found == 0:
        download_model_weights()

    app.run()
    # app.run(host='0.0.0.0', port=5000, debug=True)