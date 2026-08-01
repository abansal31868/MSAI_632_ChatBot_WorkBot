import os
from fastapi import FastAPI, Header, Request
from ingest import sync_drive_to_vector_db

app = FastAPI()


@app.post("/gdrive-webhook")
async def handle_drive_notification(
    request: Request,
    x_goog_resource_state: str = Header(None),
):
    # 'update' or 'add' states signal file modifications
    if x_goog_resource_state in ["update", "add"]:
        print("Real-time change detected in Google Drive! Syncing FAISS...")
        sync_drive_to_vector_db()
    return {"status": "success"}