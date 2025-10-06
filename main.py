# indexer/main.py
import os
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Import fungsi/variabel yang sudah ada dari processing.py
# Ini lebih efisien karena model AI tidak perlu dimuat dua kali.
from processing import process_document_in_background, get_supabase_client, model

load_dotenv()

app = FastAPI()

BUCKET = os.environ.get("STORAGE_BUCKET", "documents")

# === Schema yang sudah ada ===
class ProcessRequest(BaseModel):
    document_id: str

# === Schema BARU untuk endpoint /search ===
class SearchRequest(BaseModel):
    question: str
    top_k: int = 6
    filter_document_id: str | None = None

# === Endpoint yang sudah ada ===
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
    
    background_tasks.add_task(
        process_document_in_background,
        document_id=req.document_id,
        bucket_name=BUCKET
    )
    
    return {
        "ok": True,
        "message": "Document processing started in the background.",
        "document_id": req.document_id
    }

# ==========================================================
# ENDPOINT /SEARCH BARU YANG DITAMBAHKAN
# ==========================================================
@app.post("/search")
def search(req: SearchRequest):
    """
    Endpoint ini menerima pertanyaan, membuat embedding, dan melakukan
    pencarian kemiripan (similarity search) di database vektor.
    """
    try:
        sb = get_supabase_client()

        # 1. Buat embedding dari pertanyaan dengan prefix 'query:'
        query_embedding = model.encode(
            [f"query: {req.question}"], 
            normalize_embeddings=True, 
            show_progress_bar=False
        )[0]
        query_embedding = query_embedding.astype(np.float32).tolist()

        # 2. Panggil fungsi RPC 'match_chunks' di Supabase
        rpc_params = {
            "query_embedding": query_embedding,
            "match_count": req.top_k,
        }
        if req.filter_document_id:
            rpc_params["filter_document"] = req.filter_document_id
        
        results = sb.rpc("match_chunks", rpc_params).execute()

        if results.data:
            return {"ok": True, "items": results.data, "count": len(results.data)}
        else:
            return {"ok": True, "items": [], "count": 0}

    except Exception as e:
        print(f"Error during /search: {e}")
        raise HTTPException(status_code=500, detail=str(e))