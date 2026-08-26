"""
Utility functions for video processing, feature matching, homography calculation, 
and spatial analysis using Shapely.
"""
import os
import cv2
import numpy as np
import logging
from shapely.geometry import Polygon

# Import local configuration to access colors and other settings
import config

# Initialize logging for the utilities module
logger = logging.getLogger(__name__)


def load_labels(input_labels_file, width, height):
    """
    Reads the YOLO/custom formatted labels file and returns the scaled base polygons.
    
    Args:
        input_labels_file (str): Path to the text file containing the label data.
        width (int): Width of the video frame to scale coordinates properly.
        height (int): Height of the video frame to scale coordinates properly.
        
    Returns:
        list: A list of tuples containing (class_id, polygon_array). Returns an 
              empty list if the file cannot be read.
    """
    if not os.path.exists(input_labels_file):
        logger.error(f"Labels file not found: {input_labels_file}")
        return [] 

    try:
        with open(input_labels_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Error loading labels at {input_labels_file}: {e}")
        return []

    base_polygons = []
    for line in lines:
        data = line.strip().split()

        if len(data) >= 7:
            class_id = int(data[0])
            pts_norm = [float(val) for val in data[1:]]

            if len(pts_norm) % 2 != 0:
                continue

            pts_real = [[pts_norm[i] * width, pts_norm[i + 1] * height]
                        for i in range(0, len(pts_norm), 2)]
            
            poly_array = np.array(pts_real, dtype=np.float32).reshape((-1, 1, 2))
            base_polygons.append((class_id, poly_array))

    return base_polygons


def load_video(input_video_file, output_video_file):
    """
    Initializes video reading, prepares the output directory, and sets up the video writer.

    Args:
        input_video_file (str): Path to the source video file.
        output_video_file (str): Path where the processed video will be saved.

    Returns:
        tuple: A tuple containing (cap, out, width, height).
    """
    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)

    try:
        if not os.path.exists(input_video_file):
            raise FileNotFoundError(f"Video file not found: {input_video_file}")

        cap = cv2.VideoCapture(input_video_file)

        if not cap.isOpened():
            raise IOError(f"Could not open video at: {input_video_file}")

        logger.info(f"Video successfully loaded: {input_video_file}")

    except Exception as e:
        logger.error(f"Error loading video at {input_video_file}: {e}")
        raise

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    return cap, out, width, height


def get_homography(bf, kp_ref, des_ref, kp_curr, des_curr):
    """
    Calculates the homography matrix between the reference frame and the current frame.

    Args:
        bf (cv2.DescriptorMatcher): The feature matcher object (e.g., BFMatcher).
        kp_ref (list): Keypoints from the reference frame.
        des_ref (numpy.ndarray): Descriptors from the reference frame.
        kp_curr (list): Keypoints from the current frame.
        des_curr (numpy.ndarray): Descriptors from the current frame.

    Returns:
        numpy.ndarray or None: The 3x3 homography matrix if successful, None otherwise.
    """
    try:
        if des_ref is None or des_curr is None:
            logger.warning("Descriptors are missing. Cannot compute homography.")
            return None

        matches = bf.match(des_ref, des_curr)
        if not matches:
            logger.warning("No matches found between reference and current frame.")
            return None

        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.15)]

        if len(good_matches) < 4:
            logger.debug(f"Not enough good matches ({len(good_matches)}). Minimum 4 required.")
            return None

        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            logger.warning("cv2.findHomography returned None. Matrix calculation failed.")
            
        return M

    except Exception as e:
        logger.error(f"Unexpected error while calculating homography: {e}")
        return None


def get_transformed_polygons(base_polygons, M):
    """
    Applies the homography matrix to the base polygons and prepares them for spatial analysis.

    Args:
        base_polygons (list): A list of tuples containing (class_id, polygon_array).
        M (numpy.ndarray): The 3x3 homography matrix.

    Returns:
        list: A list of dictionaries containing 'class_id', 'cv2_poly', and 'shapely_poly'.
    """
    if M is None:
        logger.warning("Homography matrix is None. Cannot transform polygons.")
        return []

    transformed = []
    
    try:
        for class_id, poly in base_polygons:
            transformed_poly = cv2.perspectiveTransform(poly, M)
            poly_to_draw = np.int32(transformed_poly)
            shapely_poly = Polygon(poly_to_draw.reshape(-1, 2))

            if shapely_poly.is_valid:
                transformed.append({
                    "class_id": class_id,
                    "cv2_poly": poly_to_draw,
                    "shapely_poly": shapely_poly
                })
            else:
                logger.debug(f"Invalid Shapely polygon generated for class {class_id}. Skipping.")

        return transformed

    except Exception as e:
        logger.error(f"Error transforming polygons: {e}")
        return []


def evaluate_and_draw_scene(frame, transformed_polygons, autos_predictions, threshold=0.3):
    """
    Evaluates parking space occupancy and draws the results on the current frame.

    Args:
        frame (numpy.ndarray): The current video frame being processed.
        transformed_polygons (list): List of transformed polygon dictionaries.
        autos_predictions (list): Object detection predictions for vehicles.
        threshold (float): Minimum intersection percentage to consider a space occupied.
    """
    try:
        # 1. Convert all valid vehicle detections to Shapely polygons
        car_polys = []
        for pred in autos_predictions:
            try:
                x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
                
                if (x2 - x1) <= 350 and (y2 - y1) <= 350:  
                    car_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                    if car_poly.is_valid:
                        car_polys.append(car_poly)
            except Exception as e:
                logger.debug(f"Error parsing bounding box coordinates: {e}")
                continue

        # 2. Analyze each mapped polygon for occupancy
        for item in transformed_polygons:
            class_id = item["class_id"]
            cv2_poly = item["cv2_poly"]
            space_poly = item["shapely_poly"]

            if class_id == 0:
                # Driving Aisle: Keep its original configuration color
                color = config.CLASS_COLORS.get(class_id, config.DEFAULT_COLOR)
            else:
                # Parking Spaces (Regular and Handicapped)
                is_occupied = False
                for car_poly in car_polys:
                    if car_poly.area > 0:
                        try:
                            intersection_area = space_poly.intersection(car_poly).area
                            if (intersection_area / space_poly.area) > threshold:
                                is_occupied = True
                                break
                        except Exception as e:
                            logger.debug(f"Geometry intersection error: {e}")
                            continue

                # OpenCV BGR Format: Red (Occupied) = (0, 0, 255), Green (Free) = (0, 255, 0)
                color = (0, 0, 255) if is_occupied else (0, 255, 0)

            # Draw the evaluated polygon on the frame
            cv2.polylines(frame, [cv2_poly], isClosed=True, color=color, thickness=2)

    except Exception as e:
        logger.error(f"Critical error during scene evaluation and drawing: {e}")
