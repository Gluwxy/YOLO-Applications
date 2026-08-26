# Parking Space Detector 🚗🅿️

A robust computer vision pipeline for real-time parking space occupancy detection. This project combines YOLO-based object detection with advanced video stabilization and geometric analysis to accurately determine if parking spots are free or occupied, even with slight camera movements or distant vehicles.

## ✨ Features

* **Advanced Small Object Detection:** Utilizes **SAHI** (Sliced Aided Hyper Inference) alongside YOLO to accurately detect vehicles (cars, trucks, motorcycles, buses) even when they are far from the camera.
* **Camera Stabilization:** Implements **ORB feature matching** and **Homography** to dynamically adjust and stabilize the parking space polygons if the camera vibrates or shifts.
* **Precise Occupancy Evaluation:** Uses **Shapely** to calculate the exact intersection area between vehicle bounding boxes and parking space polygons, avoiding false positives.
* **Modular Architecture:** Clean, production-ready code with separated configurations, utilities, and main execution pipelines.

## 🛠️ Prerequisites

* Python 3.8 or higher
* A CUDA-compatible GPU is highly recommended for real-time processing (though CPU is supported).

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Gluwxy/YOLO-Applications.git](https://github.com/Gluwxy/YOLO-Applications.git)
