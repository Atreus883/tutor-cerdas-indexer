# indexer/chunking.py
import re
from typing import List, Dict
import pysbd

def create_chunks(text_with_pages: List[Dict], target_chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """
    Strategi chunking hibrida terbaik untuk berbagai jenis dokumen.
    Menggabungkan pemisahan berbasis paragraf dan kalimat untuk hasil yang optimal.
    """
    all_semantic_units = []
    seg = pysbd.Segmenter(language="en", clean=False)

    # Tahap 1: Pecah teks menjadi unit semantik (kalimat atau baris pendek)
    for item in text_with_pages:
        page_num = item['page']
        page_text = item.get('text', '')

        # Bersihkan spasi berlebih dan ganti baris baru ganda dengan penanda paragraf
        clean_text = re.sub(r'\s+', ' ', page_text).strip()
        paragraphs = page_text.split('\n\n') # Asumsikan paragraf dipisah oleh dua baris baru

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Gunakan pysbd untuk memecah paragraf menjadi kalimat
            sentences = seg.segment(para)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    all_semantic_units.append({'text': sent, 'page': page_num})

    # Tahap 2: Gabungkan unit semantik menjadi chunk dengan ukuran target
    chunks = []
    current_chunk_text = ""
    current_chunk_pages = set()

    for i, unit in enumerate(all_semantic_units):
        # Jika chunk kosong, mulai dengan unit saat ini
        if not current_chunk_text:
            current_chunk_text = unit['text']
            current_chunk_pages.add(unit['page'])
        # Jika menambahkan unit berikutnya tidak akan melebihi target, gabungkan
        elif len(current_chunk_text) + len(unit['text']) + 1 <= target_chunk_size:
            current_chunk_text += " " + unit['text']
            current_chunk_pages.add(unit['page'])
        # Jika chunk sudah penuh, simpan dan mulai chunk baru dengan overlap
        else:
            page_str = ", ".join(sorted([str(p) for p in current_chunk_pages]))
            chunks.append({'content': current_chunk_text, 'metadata': {'pages': page_str}})
            
            # Mulai chunk baru dengan sedikit overlap untuk menjaga konteks
            # Ambil beberapa unit terakhir dari chunk sebelumnya
            overlap_units = [u['text'] for u in all_semantic_units[max(0, i-2):i+1]]
            current_chunk_text = " ".join(overlap_units)
            current_chunk_pages = set([u['page'] for u in all_semantic_units[max(0, i-2):i+1]])

    # Jangan lupa simpan chunk terakhir
    if current_chunk_text:
        page_str = ", ".join(sorted([str(p) for p in current_chunk_pages]))
        chunks.append({'content': current_chunk_text, 'metadata': {'pages': page_str}})

    return chunks