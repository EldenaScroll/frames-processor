import os
import sys
import cv2
import numpy as np
from roboflow import Roboflow

# Roboflow config — reads API key from env var
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "crcxvzrMUhqJYcyMcpW8")
WORKSPACE = "drone-parking-management-system"
PROJECT = "drone-parking-detection"
MODEL_VERSION = 4

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
    """Enhance image for better detection, especially for dark vehicles.

    Uses LAB CLAHE for local contrast, Gamma correction for shadow recovery,
    and a subtle sharpening to define vehicle edges.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)

    kernel = np.array([[-1/9, -1/9, -1/9], [-1/9, 17/9, -1/9], [-1/9, -1/9, -1/9]])
    img = cv2.filter2D(img, -1, kernel)

    return img


def detect_cars(image, confidence=30, overlap=30, use_preprocess=True):
    """Run Roboflow detection on a cv2 image.

    Args:
        image: BGR numpy array (cv2 image)
        confidence: detection confidence threshold (0-100)
        overlap: overlap threshold (0-100)
        use_preprocess: whether to apply image enhancement

    Returns list of (x1, y1, x2, y2, confidence) tuples.
    """
    if use_preprocess:
        image = preprocess(image)

    model = _get_model()

    # Roboflow SDK can take a numpy array directly, avoiding disk I/O
    pred = model.predict(image, confidence=confidence, overlap=overlap).json()

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
    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        print(f"  [{i}] ({x1}, {y1}) -> ({x2}, {y2}) conf: {conf:.2f}")
