from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
from datetime import datetime
from absen_engine import AbsenEngine, get_db_connection

app = FastAPI(title="API Presensi cloud SMA Sjakhyakirti")
engine = AbsenEngine()
os.makedirs("hasil_presensi", exist_ok=True)

#----HALAMAN UTAMA---
@app.get("/")
def home():
    return {"status": "online", "message": "Server API Jantung AI SMA Sjakhyakirti Aktif!"}

#----BAGIAN SISWA----
@app.post("/verify-presensi")
async def verify_presensi(
    id_siswa: str = Form(...),
    nama_siswa: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    file: UploadFile = File(...)
):
    try:
        # 1. Konversi file gambar kiriman HP Android ke OpenCV Frame
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(status_code=400, content={"status": "gagal", "message": "File gambar selfie tidak valid."})

        # 2. PROSES JANTUNG AI (MTCNN & ARCFACE)
        sukses_ai, status_pesan, akurasi, koordinat = engine.verifikasi_wajah_api(img, id_siswa)

        if not sukses_ai:
            if status_pesan == "SUDAH_ABSEN":
                return JSONResponse(status_code=400, content={"status": "gagal", "message": "Anda sudah melakukan presensi hari ini!"})
            elif status_pesan == "BUKAN_PEMILIK_AKUN":
                return JSONResponse(status_code=400, content={"status": "gagal", "message": "Verifikasi Gagal: Wajah terdeteksi sebagai siswa lain!"})
            elif status_pesan == "WAJAH_TIDAK_TERDETEKSI":
                return JSONResponse(status_code=400, content={"status": "gagal", "message": "Wajah tidak terdeteksi. Posisikan kamera lebih dekat!"})
            else:
                return JSONResponse(status_code=400, content={"status": "gagal", "message": "Verifikasi Gagal: Wajah tidak cocok dengan dataset!"})

        # 3. KALKULASI GEOFENCING (RUMUS HAVERSINE)
        LAT_SEKOLAH = -2.9463   # Sesuaikan koordinat asli SMA Sjakhyakirti
        LON_SEKOLAH = 104.7571
        
        R = 6371000 # Radius bumi dalam meter
        phi1 = np.radians(latitude)
        phi2 = np.radians(LAT_SEKOLAH)
        delta_phi = np.radians(LAT_SEKOLAH - latitude)
        delta_lambda = np.radians(LON_SEKOLAH - longitude)
        
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance_geo = round(R * c, 2)

        # Batasan Radius Geofencing (Misal: 50 Meter)
        RADIUS_MAKSIMAL = 50.0
        if distance_geo > RADIUS_MAKSIMAL:
            return JSONResponse(status_code=400, content={
                "status": "gagal",
                "message": f"Gagal: Anda di luar radius sekolah! Jarak Anda: {distance_geo} meter."
            })

        # 4. BUKTI FOTO UNTUK BAB 4 SKRIPSI (OVERLAY BOX & METADATA TEXT)
        if koordinat:
            x, y, w, h = koordinat
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3) # Kotak hijau di wajah
        
        waktu_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, f"ID/Name: {id_siswa} - {nama_siswa}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"ArcFace Match: {akurasi}%", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Geo-Dist: {distance_geo} m", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, waktu_str, (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Simpan bukti file fisik JPG di server cloud
        nama_file_hasil = f"BUKTI_{id_siswa}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join("hasil_presensi", nama_file_hasil), img)

        # TODO TAHAP BERIKUTNYA BESOK: 
        # Tambahkan fungsi insert catatan_kehadiran di sini & trigger WhatsApp Gateway.

        return {
            "status": "sukses",
            "nama": nama_siswa,
            "jarak": distance_geo,
            "akurasi": akurasi,
            "pesan": "Presensi Berhasil Terverifikasi Sempurna di Cloud API!"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Terjadi kesalahan internal server: {str(e)}"})

import json

#----BAGIAN ADMIN----
@app.post("/admin/input-siswa")
async def input_siswa_baru(
    nis_siswa: str = Form(...),
    nama_siswa: str = Form(...),
    kelas_siswa: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # 1. Konversi file gambar kiriman HP Admin ke OpenCV Frame
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(status_code=400, content={"status": "gagal", "message": "File gambar master tidak valid."})

        # 2. PROSES JANTUNG AI UNTUK REGISTRASI (MTCNN & ARCFACE)
        sukses_ekstrak, vektor_wajah = engine.ekstrak_vektor_master(img)

        if not sukses_ekstrak:
            return JSONResponse(status_code=400, content={"status": "gagal", "message": "Gagal registrasi: Wajah tidak terdeteksi oleh MTCNN!"})

        # 3. MASUKKAN KE DATABASE MYSQL RAILWAY
        db = get_db_connection() # Tinggal panggil, karena sudah di-import di atas
        cursor = db.cursor()
        
        try:
            # Ubah list vektor wajah menjadi string JSON
            vektor_string = json.dumps(vektor_wajah)
            
            # id_siswa langsung diset NULL agar auto-increment berjalan sempurna
            sql = """
                INSERT INTO dataset_siswa (id_siswa, NIS, nama, jenis_kelamin, kelas, wa_ortu, face_embedding, password, role) 
                VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, 'siswa')
            """
            
            # Data default untuk kolom pelengkap database
            jenis_kelamin_default = "L/P" 
            wa_ortu_default = "-"
            password_default = str(nis_siswa) 
            
            params = (
                str(nis_siswa), 
                nama_siswa, 
                jenis_kelamin_default, 
                kelas_siswa, 
                wa_ortu_default, 
                vektor_string, 
                password_default
            )
            
            cursor.execute(sql, params)
            db.commit() 
            
            # Sinkronisasi instan RAM laptop dengan Cloud Railway
            engine.load_database_wajah()
            
        except Exception as db_err:
            db.rollback()
            return JSONResponse(status_code=400, content={"status": "gagal", "message": f"Gagal menyimpan ke database: {str(db_err)}"})
        finally:
            cursor.close()
            db.close()

        return {
            "status": "sukses",
            "nama": nama_siswa,
            "pesan": f"Siswa atas nama {nama_siswa} berhasil didaftarkan ke sistem SMA Sjakhyakirti!"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Terjadi kesalahan internal server: {str(e)}"})