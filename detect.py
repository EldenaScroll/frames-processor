import os
import sys
import tempfile
import cv2
import numpy as np
from roboflow import Roboflow

# Roboflow config — reads API key from env var
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "crcxvzrMUhqJYcyMcpW8")
WORKSPACE = "drone-parking-management-system"
PROJECT = "drone-parking-detection"
MODEL_VERSION = 3

_model = None


def _get_model():
    global _model
    if _model is None:
        if not ROBOFLOW_API_KEY:
            raise RuntimeError("ROBOFLOW_API_KEY environment variable is not set")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        _model = project.version(MODEL_VERSION).model
    return _model


def preprocess(image):
    """Enhance image for better detection in shaded areas.

    Applies Contrast Limited Adaptive Histogram Equalization
    to the lightness channel in LAB color space. This boosts local
    contrast so dark vehicles under shade become more visible without
    blowing out bright regions.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def detect_cars(image, confidence=40, overlap=30):
    """Run Roboflow detection on a cv2 image.

    Args:
        image: BGR numpy array (cv2 image)
        confidence: detection confidence threshold (0-100)
        overlap: overlap threshold (0-100)

    Returns list of (x1, y1, x2, y2, confidence) tuples.
    """
    model = _get_model()
    image = preprocess(image)

    # Roboflow SDK needs a file path, so write to a temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
        cv2.imwrite(tmp_path, image)

    try:
        pred = model.predict(tmp_path, confidence=confidence, overlap=overlap).json()
    finally:
        os.unlink(tmp_path)

    boxes = []
    for d in pred["predictions"]:
        x, y, w, h = d["x"], d["y"], d["width"], d["height"]
        conf = d.get("confidence", 0.0)
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        boxes.append((x1, y1, x2, y2, conf))

    return boxes


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect.py <image_path>")
        print("Requires ROBOFLOW_API_KEY env var to be set")
        sys.exit(1)

    image = cv2.imread(sys.argv[1])
    if image is None:
        print(f"Error: Could not load image: {sys.argv[1]}")
        sys.exit(1)

    print(f"Image: {image.shape[1]}x{image.shape[0]}")
    boxes = detect_cars(image)
    print(f"Detected {len(boxes)} vehicles:")
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        print(f"  [{i}] ({x1}, {y1}) -> ({x2}, {y2})")
