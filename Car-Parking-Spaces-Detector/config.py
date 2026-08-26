"""
Configuration module for the Parking Space Detector project.
Stores all global paths, model parameters, and visual settings.
"""
import os

# --- PATH CONFIGURATION ---
# Define the base directory of the project using the user's home folder
BASE_DIR = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'ParkingSpaceDetector')

# Path to the default input video file
DEFAULT_VIDEO_PATH = os.path.join(BASE_DIR, 'data', 'parking_trimmed.mp4')

# Path to the text file containing the polygon coordinates and classes
LABEL_FILE = os.path.join(BASE_DIR, 'data', 'labels.txt')

# Path where the processed output video will be saved
OUTPUT_VIDEO_DIR = os.path.join(BASE_DIR, 'results', 'result_parking_yolo.mp4')


# --- MODEL CONFIGURATION ---
# Filename of the YOLO model weights to be loaded by SAHI
MODEL_NAME = 'yolo26x.pt'

# Target class IDs based on the COCO dataset mapping
# 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [0, 2, 3, 5, 7]


# --- COLOR DICTIONARY (OpenCV BGR Format) ---
# Defines the colors used to draw the different polygon zones on the frame
CLASS_COLORS = {
    0: (255, 150, 0),  # 0: Driving Aisles (Light Blue/Orange)
    1: (255, 0, 0),    # 1: Handicapped Parking (Blue)
    2: (0, 255, 0),    # 2: Parking (Green)
    3: (0, 255, 255),  # 3: Pedestrian Crossing (Yellow)
}

# Fallback color used if a polygon has an unrecognized class ID
DEFAULT_COLOR = (255, 255, 255)
