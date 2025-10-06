# indexer/processing.py
import os
import io
import numpy as np
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from typing import List, Dict

from chunking import create_chunks

# Inisialisasi model dan Supabase client
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
    print(f"Starting background processing for document_id: {document_id}")

    try:
        # 1. Update status menjadi 'processing'
        sb.table("documents").update({"status": "processing"}).eq("id", document_id).execute()

        # 2. Ambil path file
        doc = sb.table("documents").select("storage_path").eq("id", document_id).single().execute().data
        if not doc or not doc.get("storage_path"):
            raise ValueError("Document path not found.")
        
        # 3. Download file
        file_bytes = sb.storage.from_(bucket_name).download(doc["storage_path"])

        # 4. Ekstrak teks halaman per halaman (filter halaman kosong)
        reader = PdfReader(io.BytesIO(file_bytes))
        text_with_pages = [
            {'page': i + 1, 'text': page.extract_text()}
            for i, page in enumerate(reader.pages)
            if (page.extract_text() or "").strip()
        ]
        
        if not text_with_pages:
            raise ValueError("No text could be extracted from any page.")

        # 5. Lakukan chunking cerdas
        chunks = create_chunks(text_with_pages)
        if not chunks:
            raise ValueError("No valid chunks were created from the document.")

        # ==========================================================
        # PERUBAHAN UTAMA: Gabungkan semua proses menjadi SATU UPSERT
        # ==========================================================
        
        # 6. Siapkan konten dari chunk yang sudah valid
        contents = [item['content'] for item in chunks]
        
        # 7. Buat embeddings dari konten tersebut
        embeddings = embed_passages(contents)
        
        # 8. Siapkan baris data lengkap (content + embedding + metadata)
        rows_to_insert = [
            {
                "document_id": document_id,
                "chunk_index": i,
                "content": contents[i],
                "embedding": embeddings[i],
                "metadata": chunks[i].get('metadata') # Ambil metadata jika ada
            }
            for i in range(len(contents))
        ]

        # 9. Hapus chunk lama dan lakukan SATU KALI upsert dengan data lengkap
        sb.table("chunks").delete().eq("document_id", document_id).execute()
        sb.table("chunks").upsert(rows_to_insert, on_conflict="document_id,chunk_index").execute()
        # ==========================================================
        
        # 10. Update status final menjadi 'embedded'
        sb.table("documents").update({
            "status": "embedded", 
            "pages": len(reader.pages)
        }).eq("id", document_id).execute()
        
        print(f"Successfully processed document_id: {document_id}")

    except Exception as e:
        print(f"Error processing document {document_id}: {e}")
        sb.table("documents").update({"status": "error"}).eq("id", document_id).execute()