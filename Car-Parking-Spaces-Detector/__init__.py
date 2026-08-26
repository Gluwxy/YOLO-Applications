"""
Parking Space Detector Package.

This package provides tools and configurations for detecting 
parking space occupancy using YOLO (via SAHI), ORB feature matching 
for video stabilization, and Shapely for geometric evaluations.
"""

# Import configuration variables to make them available at the package level
from .config import (
    BASE_DIR,
    DEFAULT_VIDEO_PATH,
    LABEL_FILE,
    OUTPUT_VIDEO_DIR,
    MODEL_NAME,
    VEHICLE_CLASSES,
    CLASS_COLORS,
    DEFAULT_COLOR
)

# Import utility functions to make them available at the package level
from .utils import (
    load_labels,
    load_video,
    get_homography,
    get_transformed_polygons,
    evaluate_and_draw_scene
)

# Define what is exported when a developer uses `from package import *`
__all__ = [
    # Config
    "BASE_DIR",
    "DEFAULT_VIDEO_PATH",
    "LABEL_FILE",
    "OUTPUT_VIDEO_DIR",
    "MODEL_NAME",
    "VEHICLE_CLASSES",
    "CLASS_COLORS",
    "DEFAULT_COLOR",
    
    # Utils
    "load_labels",
    "load_video",
    "get_homography",
    "get_transformed_polygons",
    "evaluate_and_draw_scene"
]
