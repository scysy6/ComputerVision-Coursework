# ComputerVision-Coursework
This is my Coursework of Computer Vision(COMP3065)

# Project Guidance

## Requirements

To run this program, you need to install the following Python libraries:

- `opencv-python`: For video processing and stitching.
- `torch`: For loading the YOLOv5 model.
- `numpy`: For handling numerical operations.
- `Pillow`: For working with images.

- ## Installation

Before running the code, you need to install the required libraries. You can install them by running the following command in your terminal:

```bash
pip install opencv-python torch numpy Pillow
```

## Instruction

1.Place the Video Files
Place the `.MOV` video file in the `records` folder. The program will automatically search for videos in the `records` folder and generate panoramic images in the `panorama` folder. The `panorama_detect` folder contains images  that have been corrected for rotation based on object detection.

2.Generate Panoramic Images
Run main.py to process the .MOV video files and generate panorama images:
```bash
python main.py
```

3.Automatic Cropping
After generating the panoramic image, run `autocrop.py` to automatically crop the black borders from the panorama. The new images will be saved in the `panoramas_cropped` folder.
```bash
python autocrop.py
```
