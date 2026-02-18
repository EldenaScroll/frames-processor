import os
from datetime import datetime, timezone
import requests
import cv2
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from align import align_to_master
from config import load_lot
from detect import detect_cars
from occupancy import check_occupancy

app = FastAPI()

# Required env vars
CF_BASE_URL = "https://parking.2759359719sw.workers.dev"
CF_ADMIN_TOKEN = "123456"

# Optional env vars
DEFAULT_CATEGORY = "student"
PROCESSOR_SECRET = os.environ.get("PROCESSOR_SECRET")  # optional shared secret

class ProcessReq(BaseModel):
    key: str | None = None # R2 key
    lot_id: str = "1" # default lot


def cf_query(sql: str, params: list):
    """Call Worker /query endpoint that talks to D1."""
    resp = requests.post(
        f"{CF_BASE_URL}/query",
        json={"query": sql, "params": params},
        headers={"Authorization": f"Bearer {CF_ADMIN_TOKEN}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def update_db(lot_id: str, occupied: list, free: list):
    """Update space status, confidence_score, and last_updated in D1."""
    now = datetime.now(timezone.utc).isoformat()

    for spot in occupied:
        cf_query(
            "UPDATE space SET status = ?, confidence_score = ?, last_updated = ? WHERE lot_id = ? AND id = ?",
            [1, spot["confidence"], now, lot_id, spot["id"]],
        )

    for spot in free:
        cf_query(
            "UPDATE space SET status = ?, confidence_score = ?, last_updated = ? WHERE lot_id = ? AND id = ?",
            [0, spot["confidence"], now, lot_id, spot["id"]],
        )


def load_image_by_key(key: str):
    """Download a drone image from R2 via the Worker's get-frame endpoint."""
    if not key:
        return None
    import numpy as np
    resp = requests.get(f"{CF_BASE_URL}/api/get-frame/{key}", timeout=30)
    if resp.status_code != 200:
        return None
    img_array = np.frombuffer(resp.content, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process")
def process(req: ProcessReq, request: Request):
    # Optional protection
    if PROCESSOR_SECRET:
        header = request.headers.get("x-processor-secret")
        if header != PROCESSOR_SECRET:
            raise HTTPException(status_code=401, detail="Missing/invalid processor secret")

    lot_id = str(req.lot_id)

    # Load lot config
    master_img, parking_data, err = load_lot(lot_id)
    if err:
        raise HTTPException(status_code=400, detail=err)

    # Load drone image
    image = load_image_by_key(req.key)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Could not load image: {req.key}")

    # Align to master
    align_result = align_to_master(master_img, image)
    aligned = align_result["aligned"]

    if align_result["homography"] is None:
        raise HTTPException(status_code=422, detail="Image alignment failed — not enough feature matches")

    # Detect cars
    boxes = detect_cars(aligned)

    # Check occupancy
    result = check_occupancy(boxes, parking_data)

    # Update D1 database
    update_db(lot_id, result["occupied"], result["free"])

    return {
        "success": True,
        "lot_id": lot_id,
        "spots_updated": len(parking_data),
        "occupied": len(result["occupied"]),
        "free": len(result["free"]),
    }
