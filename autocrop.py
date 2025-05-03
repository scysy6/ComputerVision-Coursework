import cv2
import numpy as np
import os
from glob import glob

def detect_black_borders(image):
    """
    Detect the width of the black borders around the image.
    :param image: Input Image
    :return: The width of the black borders in the top, right, bottom, and left directions
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    height, width = gray.shape
    
    # black threshold
    threshold = 135    
    # top
    top = 0
    for i in range(height):
        if np.mean(gray[i, :]) > threshold:
            break
        top += 1
    
    # bottom
    bottom = 0
    for i in range(height-1, -1, -1):
        if np.mean(gray[i, :]) > threshold:
            break
        bottom += 1
    
    # left
    left = 0
    for i in range(width):
        if np.mean(gray[:, i]) > threshold:
            break
        left += 1
    
    # right
    right = 0
    for i in range(width-1, -1, -1):
        if np.mean(gray[:, i]) > threshold:
            break
        right += 1
    
    # Limit the maximum cutting length and keep the bottom
    return top if top < 300 else 300, right if right < 100 else 100, bottom if bottom < 300 else 300, left if left < 100 else 100

def crop_image(image, borders):
    """
    Crop the image based on the detected black border width
    :param image: Input Image
    :param borders: Black border width in four directions (top, right, bottom, left)
    :return: Cropped image
    """
    top, right, bottom, left = borders
    height, width = image.shape[:2]
    
    # area
    crop_top = top
    crop_left = left
    crop_bottom = height - bottom
    crop_right = width - right
    
    cropped = image[crop_top:crop_bottom, crop_left:crop_right]  # crop
    
    return cropped

def process_panoramas(input_dir, output_dir):
    """
    Process all images in the panorama folder, detect and crop black edges
    :param input_dir: Input image folder path
    :param output_dir: Output image folder path
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # get all jpg files
    image_files = glob(os.path.join(input_dir, "*.jpg"))
    
    for image_file in image_files:
        print(f"Processing image: {image_file}")

        image = cv2.imread(image_file)
        
        if image is None:
            print(f"ERROR: reading {image_file} failed.")
            continue
        
        # detect black borders
        borders = detect_black_borders(image)
        print(f"Detected black border width (top, right, bottom, left): {borders}")
        
        # crop the image
        cropped_image = crop_image(image, borders)
        
        # save
        output_path = os.path.join(output_dir, os.path.basename(image_file))
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped image saved in: {output_path}")

if __name__ == "__main__":
    input_dir = "./panoramas_detect"
    output_dir = "./panoramas_cropped"
    process_panoramas(input_dir, output_dir)