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

        # 2. Ambil path file dari tabel documents
        doc = sb.table("documents").select("storage_path").eq("id", document_id).single().execute().data
        if not doc or not doc.get("storage_path"):
            raise ValueError("Document path not found.")
        
        # 3. Download file dari Supabase Storage
        file_bytes = sb.storage.from_(bucket_name).download(doc["storage_path"])

        # 4. Ekstrak teks beserta nomor halaman
        reader = PdfReader(io.BytesIO(file_bytes))
        text_with_pages = [{'page': i + 1, 'text': page.extract_text()} for i, page in enumerate(reader.pages)]
        
        # 5. Lakukan chunking cerdas
        chunks_with_metadata = create_chunks(text_with_pages)
        if not chunks_with_metadata:
            raise ValueError("No chunks were created from the document.")

        # 6. Hapus chunk lama dan simpan chunk baru
        sb.table("chunks").delete().eq("document_id", document_id).execute()
        
        rows_to_insert = [
            {"document_id": document_id, "chunk_index": i, "content": item['content']}
            for i, item in enumerate(chunks_with_metadata)
        ]
        sb.table("chunks").upsert(rows_to_insert, on_conflict="document_id,chunk_index").execute()

        # 7. Buat embeddings
        contents = [item['content'] for item in chunks_with_metadata]
        embeddings = embed_passages(contents)
        
        # 8. Simpan embeddings
        upserts_with_embeddings = [
            {"document_id": document_id, "chunk_index": i, "embedding": e}
            for i, e in enumerate(embeddings)
        ]
        sb.table("chunks").upsert(upserts_with_embeddings, on_conflict="document_id,chunk_index").execute()
        
        # 9. Update status final menjadi 'embedded'
        sb.table("documents").update({
            "status": "embedded", 
            "pages": len(reader.pages)
        }).eq("id", document_id).execute()
        
        print(f"Successfully processed document_id: {document_id}")

    except Exception as e:
        print(f"Error processing document {document_id}: {e}")
        # Jika terjadi error, update status di database
        sb.table("documents").update({"status": "error"}).eq("id", document_id).execute()