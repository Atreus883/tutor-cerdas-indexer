# indexer/Dockerfile

# Gunakan base image Python yang ringan
FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements terlebih dahulu
COPY requirements.txt .

# Install dependensi (ini akan menggunakan cache jika requirements.txt tidak berubah)
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa kode aplikasi Anda
COPY . .

# Jalankan aplikasi. Railway akan menyediakan variabel $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]