import cv2
import numpy as np
import os
from glob import glob
import torch
from PIL import Image
import warnings

# Some warning messages will be reported by torch.hub, but will not affect the operation of the program
warnings.filterwarnings("ignore")  # so lets ignore them

model = torch.hub.load('./ultralytics_yolov5_master', 'custom', path='yolov5s.pt', source='local')  # load a official model from local
model.conf = 0.15  # NMS confidence threshold
model.eval()  # inference mode

def count_objects(img):
    """
    Count the number of objects in a picture using YoloV5-s.
    :param img: numpy array of image
    :return: Number of objects
    """
    # Object detection
    results = model(img)
    detections = results.pandas().xyxy[0]
    # returns the number of detected objects
    return len(detections)


def extract_keyframes(video_path, interval=30):
    """
    Extract keyframes from video using opencv.
    :param video_path: path of video
    :param interval: frame interval
    :return: list of frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:  # save keyframe, 1/30
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    return frames

def match_features(img1, img2):
    """
    Matching feature points of two images.
    :param img1: image 1
    :param img2: image 2
    :return: Matching point pairs.
    """
    # convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # detect feature points using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # use the FLANN matcher to match feature points
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # Specify the number of KD trees = 5
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # get the coordinates of the matching points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    return pts1, pts2

def stitch_images(images):
    """
    Stitching multiple images to stitch a panorama
    :param images: list of images
    :return: stitched panorama
    """
    if len(images) < 2:  # bottom-line strategy, not be triggered
        return images[0]
    
    # Using OpenCV stitcher
    stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(images)
    
    if status != cv2.Stitcher_OK:
        print("ERROR: Stitching failed")
        panorama = None
    
    return panorama

def process_videos(input_dir, output_dir):
    """
    Process all .MOVs in a input_dir, and generate panoramas and save them in output_dir.
    Especially, detected results will save in {ouput_dir}_detect.
    :param input_dir: directory path of inputs
    :param output_dir: directory path of oupts
    """
    assert os.path.exists(input_dir), "input_dir not exist!"  # check input_dir

    if not os.path.exists(output_dir):  # check output_dir
        os.makedirs(output_dir)

    output_dir_detect = f"{output_dir}_detect"
    if not os.path.exists(output_dir_detect):  # check {output_dir}_detect
        os.makedirs(output_dir_detect)
    
    video_files = glob(os.path.join(input_dir, "*.MOV"))
    
    for video_file in video_files:
        print(f"Processing: {video_file}")
        # extract keyframes
        frames = extract_keyframes(video_file)
        # generate panorama
        panorama = stitch_images(frames)
        output_path = os.path.join(output_dir, os.path.basename(video_file).replace(".MOV", ".jpg").replace(".mov", ".jpg"))
        cv2.imwrite(output_path, panorama)  # save the detectedd result
        print(f"Saved original result: {output_path}")

        # detect rotation via object detection
        panorama_rotate = cv2.rotate(panorama, cv2.ROTATE_180)
        objects_num = count_objects(panorama)
        objects_num_rotate = count_objects(panorama_rotate)

        print(f"Number of objects detected: {objects_num}, Number of objects detected after rotation: {objects_num_rotate}")
        # if there are more objects after rotation, rotate the panorama 180 degrees
        if objects_num_rotate > objects_num:
            panorama = panorama_rotate

        # save results
        output_path = os.path.join(output_dir_detect, os.path.basename(video_file).replace(".MOV", ".jpg").replace(".mov", ".jpg"))
        cv2.imwrite(output_path, panorama)  # save the detected result
        print(f"Saved result after rotation correction: {output_path}")


if __name__ == "__main__":
    input_dir = "./records"
    output_dir = "./panoramas"
    process_videos(input_dir, output_dir)