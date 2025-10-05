# Langkah 1: Pilih "sistem operasi" dasar. Kita pilih Python versi 3.11.
FROM python:3.11-slim

# Langkah 2: Instal semua perkakas sistem yang kita butuhkan.
# Perintah ini akan berjalan saat "image" kita dibuat, bukan saat aplikasi dijalankan.
# Di sinilah kita menyelesaikan masalah 'read-only file system'.
RUN apt-get update && apt-get install -y build-essential gcc python3-dev

# Langkah 3: Tentukan folder kerja di dalam container
WORKDIR /app

# Langkah 4: Salin file requirements.txt terlebih dahulu
COPY requirements.txt .

# Langkah 5: Install semua package Python dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Langkah 6: Salin semua sisa kode proyek Anda ke dalam container
COPY . .

# Langkah 7: Perintah untuk menjalankan aplikasi Anda saat container启动
# Ganti 'main:app' dan port '8000' jika perlu, sesuaikan dengan proyek Anda
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
