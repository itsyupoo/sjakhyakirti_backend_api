# 1. Gunakan basis Linux Python resmi yang super stabil dan ringan
FROM python:3.10-slim

# 2. Setel folder kerja di dalam server
WORKDIR /app

# 3. Instal langsung paket biner tameng OpenCV (Anti-Eror libxcb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Salin daftar library dan instal semuanya
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Salin seluruh kodingan skripsi kamu ke dalam server
COPY . .

# 6. Jalankan FastAPI menggunakan Uvicorn secara resmi
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8080"]