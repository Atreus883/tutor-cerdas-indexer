# indexer/main.py
import os
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

from processing import process_document_in_background

load_dotenv()

app = FastAPI()

BUCKET = os.environ.get("STORAGE_BUCKET", "documents")

class ProcessRequest(BaseModel):
    document_id: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/process")
async def process_document(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Endpoint ini menerima document_id, langsung merespons, dan memproses
    dokumen di latar belakang untuk pengalaman pengguna yang lebih baik.
    """
    print(f"Received request to process document_id: {req.document_id}")
    
    # Menambahkan tugas berat ke background
    background_tasks.add_task(
        process_document_in_background,
        document_id=req.document_id,
        bucket_name=BUCKET
    )
    
    # Langsung kembalikan respons agar pengguna tidak menunggu
    return {
        "ok": True,
        "message": "Document processing started in the background.",
        "document_id": req.document_id
    }