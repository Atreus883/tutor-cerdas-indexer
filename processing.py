# indexer/processing.py (Versi dengan Logging Diagnostik)
import os
import tempfile
import numpy as np
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from typing import List
import json # Tambahkan import json untuk print yang lebih rapi

from chunking import chunk_document_by_layout

MODEL_NAME = os.environ.get("EMBED_MODEL", "intfloat/e5-small-v2")
model = SentenceTransformer(MODEL_NAME)

def get_supabase_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)

def embed_passages(texts: List[str]) -> List[List[float]]:
    texts_with_prefix = [f"passage: {t or ''}" for t in texts]
    vecs = model.encode(texts_with_prefix, normalize_embeddings=True, batch_size=32)
    return vecs.astype(np.float32).tolist()

def process_document_in_background(document_id: str, bucket_name: str):
    sb = get_supabase_client()
    print(f"--- Starting ADVANCED processing for document_id: {document_id} ---")

    try:
        sb.table("documents").update({"status": "processing"}).eq("id", document_id).execute()

        doc = sb.table("documents").select("storage_path, title").eq("id", document_id).single().execute().data
        if not doc or not doc.get("storage_path"):
            raise ValueError("Document path not found.")
        
        file_bytes = sb.storage.from_(bucket_name).download(doc["storage_path"])

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
            temp_pdf.write(file_bytes)
            temp_pdf.flush()
            
            print("--- Step 1: Running layout-aware chunking with 'unstructured' ---")
            chunks = chunk_document_by_layout(temp_pdf.name)
            print(f"--- Step 2: Chunking complete. Found {len(chunks)} potential chunks. ---")

        # ==========================================================
        # DEBUGGING: Cetak isi mentah dari variabel chunks
        # ==========================================================
        print("--- DEBUG: Raw chunks data before filtering and embedding: ---")
        # Menggunakan json.dumps agar outputnya lebih mudah dibaca di log
        print(json.dumps(chunks, indent=2))
        print("--- END DEBUG ---")
        # ==========================================================

        if not chunks:
            raise ValueError("No valid chunks could be created. The document might be empty or unparsable.")

        contents = [item['content'] for item in chunks if (item.get('content') or "").strip()]
        if not contents:
            raise ValueError("After filtering, no chunks with valid content were found.")
        
        print(f"--- Step 3: Filtered down to {len(contents)} valid chunks. Starting embedding. ---")
        embeddings = embed_passages(contents)
        print("--- Step 4: Embedding complete. Preparing data for database. ---")
        
        # Rekonstruksi baris data HANYA dari chunk yang valid
        valid_chunks = [item for item in chunks if (item.get('content') or "").strip()]
        rows_to_insert = [
            {
                "document_id": document_id,
                "chunk_index": i,
                "content": valid_chunks[i]['content'],
                "embedding": embeddings[i],
                "metadata": valid_chunks[i]['metadata']
            }
            for i in range(len(valid_chunks))
        ]

        sb.table("chunks").delete().eq("id", document_id).execute()
        sb.table("chunks").upsert(rows_to_insert, on_conflict="document_id,chunk_index").execute()
        print("--- Step 5: Successfully upserted chunks to database. ---")
        
        sb.table("documents").update({
            "status": "embedded", 
            "pages": valid_chunks[-1]['metadata']['pages'].split(',')[-1].strip() if valid_chunks else 0
        }).eq("id", document_id).execute()
        
        print(f"--- Successfully processed document_id: {document_id} with {len(valid_chunks)} chunks. ---")

    except Exception as e:
        print(f"--- ERROR during processing document {document_id}: {e} ---")
        sb.table("documents").update({"status": "error"}).eq("id", document_id).execute()