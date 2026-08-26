"""
Main execution pipeline for the Parking Space Detector.
Utilizes SAHI for sliced YOLO inference, ORB-based homography for camera
stabilization, and Shapely polygon intersections for parking spot occupancy tracking.
"""
import cv2
import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Direct imports from local modules
from config import (
    DEFAULT_VIDEO_PATH,
    OUTPUT_VIDEO_DIR,
    LABEL_FILE,
    MODEL_NAME,
    VEHICLE_CLASSES
)
from utils import (
    load_video,
    load_labels,
    get_homography,
    get_transformed_polygons,
    evaluate_and_draw_scene
)

# Initialize logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Executes the video processing pipeline:
    1. Loads the detection model and video stream.
    2. Extracts base annotation polygons.
    3. Stabilizes each frame using ORB feature matching and homography.
    4. Runs sliced prediction on moving frames and evaluates spatial occupancy.
    5. Writes the annotated frames to the output file.
    """
    logger.info("Loading YOLO model via SAHI...")

    cap = None
    out = None

    try:
        # Load the detection model using SAHI with parameters from config
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=MODEL_NAME,
            confidence_threshold=0.25,
            device="cuda:0",  # Change to "cpu" if running without GPU support
        )

        # 1. Video I/O setup
        logger.info("Initializing video stream and writer...")
        cap, out, width, height = load_video(DEFAULT_VIDEO_PATH, OUTPUT_VIDEO_DIR)

        # 2. Load ground-truth polygons scaled to video dimensions
        logger.info("Loading base labels from configuration...")
        base_polygons = load_labels(LABEL_FILE, width, height)
        
        if not base_polygons:
            logger.warning("No base polygons were loaded. Occupancy overlay will be skipped.")

        # 3. Initialize ORB and Matcher for camera motion stabilization
        orb = cv2.ORB_create(nfeatures=2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Extract keypoints and descriptors from the initial reference frame
        ret, frame_ref = cap.read()
        if not ret:
            logger.error("Failed to read the initial reference frame. Terminating.")
            return

        gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
        kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)

        logger.info("Starting frame processing loop...")

        # 4. Process video frame by frame
        frame_idx = 0
        while cap.isOpened():
            ret, frame_curr = cap.read()
            if not ret:
                logger.info("Reached end of video stream.")
                break

            frame_idx += 1
            gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
            kp_curr, des_curr = orb.detectAndCompute(gray_curr, None)

            # Compute homography relative to reference frame
            M = get_homography(bf, kp_ref, des_ref, kp_curr, des_curr)

            if M is not None:
                # Transform base polygons to match current frame perspective
                transformed_polygons = get_transformed_polygons(base_polygons, M)

                # Run sliced inference for improved small-object resolution
                result = get_sliced_prediction(
                    frame_curr,
                    detection_model,
                    slice_height=512,
                    slice_width=512,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2
                )

                # Filter detections based on configured vehicle class IDs
                autos_predictions = [
                    pred for pred in result.object_prediction_list
                    if pred.category.id in VEHICLE_CLASSES
                ]

                # Evaluate polygon intersections and draw bounding areas
                evaluate_and_draw_scene(
                    frame=frame_curr,
                    transformed_polygons=transformed_polygons,
                    autos_predictions=autos_predictions,
                    threshold=0.3
                )
            else:
                logger.warning(f"Tracking lost on frame {frame_idx}: insufficient feature matches.")

            # Write annotated frame to output video
            out.write(frame_curr)

        logger.info(f"Processing complete. Output saved to: {OUTPUT_VIDEO_DIR}")

    except Exception as e:
        logger.error(f"Execution error encountered: {e}")

    finally:
        # 5. Clean up OpenCV resources safely
        logger.info("Releasing video handles and closing windows...")
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
