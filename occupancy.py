import cv2
import numpy as np


def check_occupancy(boxes, parking_data):
    """Determine which parking spots are occupied by detected vehicles.

    Args:
        boxes: list of (x1, y1, x2, y2) bounding box tuples
        parking_data: list of (polygon_ndarray, spot_id) tuples

    Returns dict with:
        occupied: list of spot IDs that have a vehicle
        free: list of spot IDs with no vehicle
    """
    occupied = []
    free = []

    for polygon, spot_id in parking_data:
        is_occupied = False
        for (x1, y1, x2, y2) in boxes:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                is_occupied = True
                break

        if is_occupied:
            occupied.append(spot_id)
        else:
            free.append(spot_id)

    return {"occupied": occupied, "free": free}


if __name__ == "__main__":
    # Quick test with fake data
    test_polygon = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.int32)
    parking_data = [
        (test_polygon, "A1"),
        (np.array([[300, 300], [400, 300], [400, 400], [300, 400]], np.int32), "A2"),
    ]

    # Box center (150, 150) is inside A1, nothing inside A2
    boxes = [(120, 120, 180, 180)]

    result = check_occupancy(boxes, parking_data)
    print(f"Occupied: {result['occupied']}")
    print(f"Free:     {result['free']}")

    assert result["occupied"] == ["A1"], f"Expected ['A1'], got {result['occupied']}"
    assert result["free"] == ["A2"], f"Expected ['A2'], got {result['free']}"
