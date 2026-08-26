"""
This script implements a robust video processing pipeline for parking space occupancy detection. 
It leverages a YOLO model via SAHI (Slicing Aided Hyper Inference) for high-accuracy vehicle 
detection, and uses OpenCV's ORB (Oriented FAST and Rotated BRIEF) feature matching and homography to stabilize predefined 
parking space polygons against camera movement. By evaluating the intersection between the 
detected cars and the stabilized zones, it determines parking availability frame-by-frame 
and exports an annotated output video.
"""


import cv2
import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Import local modules from the same directory
import config
import utils

# Initialize logging for the main execution script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Main execution pipeline for parking space occupancy detection.
    Initializes the object detection model, processes the video frame by frame,
    applies homography for stabilization, and evaluates parking space occupancy.
    """
    logger.info("Loading YOLO model via SAHI...")

    # Initialize variables to None to ensure safe cleanup in the 'finally' block
    cap = None
    out = None

    try:
        # Load the detection model using SAHI
        # Ensure 'yolo26x.pt' exists in your environment
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path="yolo26x.pt",
            confidence_threshold=0.25,
            device="cuda:0",  # Change to "cpu" if no GPU is available
        )

        # 1. Video I/O Configuration
        logger.info("Initializing video reading and writing...")
        cap, out, width, height = utils.load_video(config.DEFAULT_VIDEO_PATH, config.OUTPUT_VIDEO_DIR)

        # 2. Load base labels (parking space polygons)
        logger.info("Loading base polygons from labels file...")
        base_polygons = utils.load_labels(config.LABEL_FILE, width, height)

        if not base_polygons:
            logger.warning("No base polygons loaded. Processing will continue, but no zones will be drawn.")

        # 3. Initial ORB Configuration for feature matching
        orb = cv2.ORB_create(nfeatures=2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Read the first frame to use as the reference frame for homography
        ret, frame_ref = cap.read()
        if not ret:
            logger.error("Failed to read the first frame. Exiting process.")
            return

        gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
        kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)

        logger.info('Processing video with SAHI and Shapely for occupancy detection...')

        # 4. Main video processing loop
        frame_count = 0
        while cap.isOpened():
            ret, frame_curr = cap.read()
            if not ret:
                logger.info("End of video stream reached.")
                break

            frame_count += 1
            gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
            kp_curr, des_curr = orb.detectAndCompute(gray_curr, None)

            # Compute homography to stabilize the current frame relative to the reference frame
            M = utils.get_homography(bf, kp_ref, des_ref, kp_curr, des_curr)

            if M is not None:
                # Get stabilized polygons for the current frame
                transformed_polygons = utils.get_transformed_polygons(base_polygons, M)

                # Perform sliced inference with SAHI
                result = get_sliced_prediction(
                    frame_curr,
                    detection_model,
                    slice_height=512,
                    slice_width=512,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2
                )

                # Filter predictions to keep only cars (assuming category ID 2 is 'car' in COCO)
                autos_predictions = [
                    pred for pred in result.object_prediction_list
                    if pred.category.id == 2
                ]

                # Evaluate intersections and draw the scene
                utils.evaluate_and_draw_scene(frame_curr, transformed_polygons, autos_predictions, threshold=0.3)
            else:
                logger.warning(f"Tracking lost in frame {frame_count} (too few matching points).")

            # Write the processed frame to the output video file
            out.write(frame_curr)

        logger.info(f"Video successfully processed and saved to: {config.OUTPUT_VIDEO_DIR}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}")

    finally:
        # 5. Resource Cleanup
        # This block executes regardless of whether the try block succeeds or throws an error
        logger.info("Releasing resources and cleaning up...")
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
