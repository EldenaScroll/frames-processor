import cv2
import numpy as np


def check_occupancy(boxes, parking_data, overlap_threshold=0.3):
    """Determine which parking spots are occupied by detected vehicles using area overlap.

    Args:
        boxes: list of (x1, y1, x2, y2, confidence) tuples
        parking_data: list of (polygon_ndarray, spot_id) tuples
        overlap_threshold: minimum percentage (0.0 to 1.0) of a spot's area 
                           that must be covered by a car to count as occupied.

    Returns dict with:
        occupied: list of {"id": spot_id, "confidence": float}
        free: list of {"id": spot_id, "confidence": 0.0}
    """
    occupied = []
    free = []

    for polygon, spot_id in parking_data:
        px, py, pw, ph = cv2.boundingRect(polygon)
        
        if pw <= 0 or ph <= 0:
            free.append({"id": spot_id, "confidence": 0.0})
            continue

        spot_mask = np.zeros((ph, pw), dtype=np.uint8)
        cv2.fillPoly(spot_mask, [polygon - [px, py]], 255)
        spot_area = np.count_nonzero(spot_mask)

        best_conf = 0.0
        max_overlap = 0.0
        is_occupied = False

        for box in boxes:
            bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            conf = box[4] if len(box) > 4 else 0.0

            ix1 = max(0, bx1 - px)
            iy1 = max(0, by1 - py)
            ix2 = min(pw, bx2 - px)
            iy2 = min(ph, by2 - py)

            if ix2 > ix1 and iy2 > iy1:
                intersection_mask = spot_mask[iy1:iy2, ix1:ix2]
                overlap_area = np.count_nonzero(intersection_mask)
                overlap_ratio = overlap_area / spot_area

                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio
                
                if overlap_ratio >= overlap_threshold:
                    is_occupied = True
                    best_conf = max(best_conf, conf)

        if is_occupied:
            occupied.append({"id": spot_id, "confidence": best_conf})
        else:
            free.append({"id": spot_id, "confidence": 0.0})

    return {"occupied": occupied, "free": free}


if __name__ == "__main__":
    test_polygon = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.int32)
    parking_data = [
        (test_polygon, "A1"),
        (np.array([[300, 300], [400, 300], [400, 400], [300, 400]], np.int32), "A2"),
    ]

    boxes = [(120, 120, 180, 180, 0.95)]

    result = check_occupancy(boxes, parking_data)
    print(f"Occupied: {result['occupied']}")
    print(f"Free:     {result['free']}")

    assert result["occupied"] == [{"id": "A1", "confidence": 0.95}], f"Unexpected occupied: {result['occupied']}"
    assert result["free"] == [{"id": "A2", "confidence": 0.0}], f"Unexpected free: {result['free']}"
    print("All tests passed!")

