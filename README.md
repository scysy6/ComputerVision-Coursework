# ComputerVision-Coursework
This is my Coursework of Computer Vision(COMP3065)

# Project Guidance

## Project Structure
The project structure is as follows:
```
├── main.py                    # Main script for processing videos and generating panoramas
├── autocrop.py                # Script for automatically cropping black borders from panoramas
├── records/                   # Input folder for .mov video files
├── panoramas/                 # Output folder for generated panorama images
├── panoramas_detect/          # Output folder for object detection and rotation-corrected panoramas
├── panoramas_cropped/         # Output folder for cropped panoramas
├── ultralytics_yolov5_master/ # Folder containing the YOLOv5 model files (local repository)
```


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

## YOLOv5 Model Loading
This project uses the YOLOv5 model for object detection, which is loaded through the `torch.hub.load` function from the local path.

In the code, the following line is used to load the model:
```bash
model = torch.hub.load('./ultralytics_yolov5_master', 'custom', path='yolov5s.pt', source='local')
```
The model (yolov5s.pt) is loaded from the `ultralytics_yolov5_master folder`, which contains the official YOLOv5 repository and pre-trained weights.


## Instruction

1.Place the Video Files
Place the `.MOV` video file in the `records` folder. The program will automatically search for videos in the `records` folder and generate panoramic images in the `panorama` folder. The `panorama_detect` folder contains images  that have been corrected for rotation based on object detection.

2.Generate Panoramic Images
Run main.py to process the .mov video files and generate panorama images:
```bash
python main.py
```

3.Automatic Cropping
After generating the panoramic image, run `autocrop.py` to automatically crop the black borders from the panorama. The new images will be saved in the `panoramas_cropped` folder.
```bash
python autocrop.py
```






