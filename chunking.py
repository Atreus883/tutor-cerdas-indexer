# indexer/chunking.py
import pysbd
from typing import List, Dict

def chunk_text_with_metadata(text_with_pages: List[Dict]) -> List[Dict]:
    """
    Menerima daftar teks per halaman, memecahnya menjadi kalimat, lalu
    menggabungkannya menjadi chunk yang lebih besar dengan metadata halaman.
    """
    seg = pysbd.Segmenter(language="en", clean=False)
    
    all_sentences_with_pages = []
    for item in text_with_pages:
        page_num = item['page']
        page_text = item['text']
        sentences = seg.segment(page_text)
        for sent in sentences:
            all_sentences_with_pages.append({'sentence': sent, 'page': page_num})
            
    chunks = []
    current_chunk = ""
    chunk_pages = set()
    
    target_chunk_size = 1000  # Karakter per chunk (bisa disesuaikan)
    overlap_sentences = 2 # Jumlah kalimat tumpang tindih untuk konteks

    i = 0
    while i < len(all_sentences_with_pages):
        sentence_data = all_sentences_with_pages[i]
        sentence_text = sentence_data['sentence']
        page_num = sentence_data['page']

        if len(current_chunk) + len(sentence_text) <= target_chunk_size:
            current_chunk += " " + sentence_text
            chunk_pages.add(page_num)
            i += 1
        else:
            # Chunk sudah penuh, simpan dan siapkan chunk baru dengan overlap
            chunk_page_str = ", ".join(sorted([str(p) for p in chunk_pages]))
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': {'pages': chunk_page_str}
            })
            
            # Reset untuk chunk berikutnya dengan overlap
            overlap_start_index = max(0, i - overlap_sentences)
            current_chunk = " ".join([s['sentence'] for s in all_sentences_with_pages[overlap_start_index:i]])
            chunk_pages = set([s['page'] for s in all_sentences_with_pages[overlap_start_index:i]])

    # Simpan sisa chunk terakhir
    if current_chunk:
        chunk_page_str = ", ".join(sorted([str(p) for p in chunk_pages]))
        chunks.append({
            'content': current_chunk.strip(),
            'metadata': {'pages': chunk_page_str}
        })
        
    return chunks