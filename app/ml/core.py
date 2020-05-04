import imageio
from skimage.transform import resize
import cv2
from .demo import load_checkpoints
from .demo import make_animation
from skimage import img_as_ubyte
import time
from google.cloud import storage
import os
import random
import string

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

VIDEO_FOLDER_NAME = 'videos'
PHOTOS_FOLDER_NAME = 'photos'
PROCESSED_FOLDER_NAME = 'processed'
CLOUD_STORAGE_BUCKET = 'heroic-gantry-275322.appspot.com'

def process(video, image):

    # Start time
    start_time = time.time()
    
    # Video
    fps_of_video = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS))
    frames_of_video = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))

    # Photo
    source_image = imageio.imread(image)
    source_image = resize(source_image, (256, 256))[..., :3]

    driving_video = imageio.mimread(video, memtest=False)
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    generator, kp_detector = load_checkpoints(config_path=os.path.join(THIS_FOLDER, 'config/vox-256.yaml'), checkpoint_path=os.path.join(THIS_FOLDER, 'vox-cpk.pth.tar'))

    # TODO: Clean this up to a shared instance

    # Create a Cloud Storage client.
    gcs = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)

    # New file to be uploaded
    new_file_name = randomString(10) + ".mp4"

    print("Starting the transformation...")
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, cpu=True)
    imageio.mimsave(new_file_name, [img_as_ubyte(frame) for frame in predictions])
    print("Finished transformation...")

    print("Cleaning up...")
    os.remove(video)
    os.remove(image)
    print("Cleanup Finished.")

    blob = bucket.blob(new_file_name)
    blob.upload_from_filename(new_file_name)
    # blob.make_public()

    # FULL_FILE_PATH = f"gs://{CLOUD_STORAGE_BUCKET}/{PROCESSED_FOLDER_NAME}/{new_file_name}"

    # #open filestream with write permissions
    # with open(new_file_name, mode="wb") as downloaded_file:
    #     #download and write file locally 
    #     gcs.download_blob_to_file(FULL_FILE_PATH, downloaded_file)

    print("--- %s seconds ---" % (time.time() - start_time))
    
    return blob.public_url

def randomString(stringLength):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(stringLength))
