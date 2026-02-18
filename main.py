import os
import requests
import datetime
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Required env vars
CF_BASE_URL = "https://parking.2759359719sw.workers.dev"
CF_ADMIN_TOKEN = "123456"

# Optional env vars
SPACE_ID = "Space_1"
DEFAULT_CATEGORY = "student"
PROCESSOR_SECRET = os.environ.get("PROCESSOR_SECRET")  # optional shared secret

class ProcessReq(BaseModel):
    key: str | None = None      # upload key (not used yet)
    lot_id: str = "1"           # default lot

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

    # ensure Space_1 row exists
    exists = cf_query(
        "SELECT 1 FROM space WHERE lot_id = ? AND id = ? LIMIT 1;",
        [lot_id, SPACE_ID]
    )
    has_row = bool(exists.get("results"))

    if not has_row:
        cf_query(
            "INSERT INTO space (id, lot_id, category, status) VALUES (?, ?, ?, ?);",
            [SPACE_ID, lot_id, DEFAULT_CATEGORY, 0]
        )

    # update status to 1
    out = cf_query(
        "UPDATE space SET status = 1, last_updated = ? WHERE lot_id = ? AND id = ?;",
        [str(datetime.datetime.now()),lot_id, SPACE_ID]
    )

    return {"success": True, "space_id": SPACE_ID, "lot_id": lot_id, "update_result": out}
