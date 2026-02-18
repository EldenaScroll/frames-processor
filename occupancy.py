import cv2
import numpy as np


def check_occupancy(boxes, parking_data):
    """Determine which parking spots are occupied by detected vehicles.

    Args:
        boxes: list of (x1, y1, x2, y2, confidence) tuples
        parking_data: list of (polygon_ndarray, spot_id) tuples

    Returns dict with:
        occupied: list of {"id": spot_id, "confidence": float}
        free: list of {"id": spot_id, "confidence": 0.0}
    """
    occupied = []
    free = []

    for polygon, spot_id in parking_data:
        best_conf = 0.0
        is_occupied = False
        for box in boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            conf = box[4] if len(box) > 4 else 0.0
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                is_occupied = True
                best_conf = max(best_conf, conf)

        if is_occupied:
            occupied.append({"id": spot_id, "confidence": best_conf})
        else:
            free.append({"id": spot_id, "confidence": 0.0})

    return {"occupied": occupied, "free": free}


if __name__ == "__main__":
    # Quick test with fake data
    test_polygon = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.int32)
    parking_data = [
        (test_polygon, "A1"),
        (np.array([[300, 300], [400, 300], [400, 400], [300, 400]], np.int32), "A2"),
    ]

    # Box center (150, 150) is inside A1, nothing inside A2
    boxes = [(120, 120, 180, 180, 0.95)]

    result = check_occupancy(boxes, parking_data)
    print(f"Occupied: {result['occupied']}")
    print(f"Free:     {result['free']}")

    assert result["occupied"] == [{"id": "A1", "confidence": 0.95}], f"Unexpected occupied: {result['occupied']}"
    assert result["free"] == [{"id": "A2", "confidence": 0.0}], f"Unexpected free: {result['free']}"
    print("All tests passed!")

