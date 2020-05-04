from app import app
import os
import requests
from flask import jsonify, request, logging
from google.cloud import storage
import tempfile
import cv2
from .ml import process

VIDEO_FOLDER_NAME = 'videos'
PHOTOS_FOLDER_NAME = 'photos'
CLOUD_STORAGE_BUCKET = 'heroic-gantry-275322.appspot.com'

@app.route('/')
def index():
    return 'Make Face Ready'

@app.route('/handle_processing', methods=['POST'])
def handle_processing():

    # Get json body
    json_body = request.get_json()
    video = json_body['video_name']
    photo = json_body['photo_name']

    # Create a Cloud Storage client.
    gcs = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)

    # Fetch files and create temporary file to be used
    video_blob = bucket.blob(VIDEO_FOLDER_NAME + "/" + video)
    video_blob.download_to_filename(video)

    photo_blob = bucket.blob(PHOTOS_FOLDER_NAME + "/" + photo)
    photo_blob.download_to_filename(photo)
    photo_blob.make_public()

    p = process(video, photo)

    # The public URL can be used to directly access the uploaded file via HTTP.
    return jsonify(p)

# @app.errorhandler(500)
# def server_error(e):
#     logging.exception('An error occurred during a request.')
#     return """
#     An internal error occurred: <pre>{}</pre>
#     See logs for full stacktrace.
#     """.format(e), 500

