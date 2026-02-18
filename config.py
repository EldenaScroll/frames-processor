import os
import json
import cv2
import numpy as np

LOT_DIR = "lots"


def load_lot(lot_id):
    """Load master image and parking spot definitions for a lot.

    Returns (master_img, parking_data, error_string).
    parking_data is a list of (polygon_ndarray, spot_id) tuples,
    matching the format used by the occupancy checker.
    """
    lot_path = os.path.join(LOT_DIR, lot_id)
    master_path = os.path.join(lot_path, "master.jpg")
    parking_path = os.path.join(lot_path, "parking.json")

    # Try .jpg first, then .JPG
    if not os.path.exists(master_path):
        master_path = os.path.join(lot_path, "master.JPG")
    if not os.path.exists(master_path):
        return None, None, f"Master image missing for {lot_id}"

    if not os.path.exists(parking_path):
        return None, None, f"Parking data missing for {lot_id}"

    master_img = cv2.imread(master_path)
    if master_img is None:
        return None, None, f"Failed to decode master image for {lot_id}"

    with open(parking_path, "r") as f:
        spots_json = json.load(f)

    # Convert back to the (polygon_ndarray, spot_id) format
    parking_data = []
    for spot in spots_json:
        polygon = np.array(spot["polygon"], np.int32)
        parking_data.append((polygon, spot["id"]))

    return master_img, parking_data, None


if __name__ == "__main__":
    img, data, err = load_lot("lotA")
    if err:
        print(f"Error: {err}")
    else:
        print(f"Master image: {img.shape[1]}x{img.shape[0]}")
        print(f"Loaded {len(data)} parking spots:")
        for polygon, spot_id in data:
            print(f"  {spot_id}: {len(polygon)} points")
