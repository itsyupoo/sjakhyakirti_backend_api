import numpy as np
import os
import json
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
from datetime import datetime
from absen_engine import AbsenEngine, get_db_connection
from pydantic import BaseModel
from mysql.connector import Error
from zoneinfo import ZoneInfo
from uuid import uuid4
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def ambil_pengaturan_geofencing():
    db = get_db_connection()
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT latitude, longitude, radius, template_wa
            FROM konfigurasi_geofencing
            LIMIT 1
        """)
        data = cursor.fetchone()
        cursor.close()
        db.close()
        return data
    except Exception as e:
        print(f"ERROR GEOFENCING: {e}")
        return {
            "latitude": -2.994583,
            "longitude": 104.756111,
            "radius": 100,
            "template_wa": ""
        }

app = FastAPI(title="API Presensi cloud SMA Sjakhyakirti")
engine = AbsenEngine()
os.makedirs("hasil_presensi", exist_ok=True)

def kirim_wa_fonnte(
    nomor,
    pesan,
    path_foto
):
    try:
        headers = {"Authorization":os.getenv("FONNTE_TOKEN")}
        data = {"target": nomor,"message": pesan,}
        with open(path_foto, "rb") as foto:
            files = {"file": foto}
            response = requests.post("https://api.fonnte.com/send",headers=headers,data=data,files=files,timeout=30)
        print("RESPON FONNTE:",response.text)
        return True
    except Exception as e:
        print(
            "ERROR FONNTE:",e)
        return False
    
def upload_ke_google_drive(path_foto):
    try:
        print("=== MASUK GOOGLE DRIVE ===")

        print(
            "GOOGLE JSON ADA:",
            os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") is not None
        )
        
        service_account_info = json.loads(
            os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        )

        print(
            "SERVICE ACCOUNT:",
            service_account_info.get("client_email")
        )

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/drive"]
        )

        service = build(
            "drive",
            "v3",
            credentials=credentials
        )

        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        print("FOLDER ID:", folder_id)

        file_metadata = {
            "name": os.path.basename(path_foto),
            "parents": [folder_id]
        }

        media = MediaFileUpload(
            path_foto,
            resumable=False
        )

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id"
        ).execute()

        file_id = file.get("id")

        # Buat file bisa dilihat publik
        service.permissions().create(
            fileId=file_id,
            body={
                "type": "anyone",
                "role": "reader"
            }
        ).execute()

        link = f"https://drive.google.com/file/d/{file_id}/view"

        print("GOOGLE DRIVE URL:", link)

        return link

    except Exception as e:
        import traceback
        traceback.print_exc()

        print("ERROR GOOGLE DRIVE:", e)
        return None
        
def hitung_jarak(
    latitude,
    longitude,
    lat_sekolah,
    lon_sekolah
):
    R = 6371000
    phi1 = np.radians(latitude)
    phi2 = np.radians(lat_sekolah)
    delta_phi = np.radians(
        lat_sekolah - latitude
    )

    delta_lambda = np.radians(
        lon_sekolah - longitude
    )

    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1)
        * np.cos(phi2)
        * np.sin(delta_lambda / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return round(R * c, 2)

async def upload_to_cv2(file):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )
    if img is None:
        return None
    # Resize jika terlalu besar
    h, w = img.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img = cv2.resize(
            img,
            (
                int(w * scale),
                int(h * scale)
            )
        )
    print(f"Ukuran gambar: {img.shape}")
    return img

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
        img = await upload_to_cv2(file)
        
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

        data_geo = ambil_pengaturan_geofencing()
        LAT_SEKOLAH = float(data_geo["latitude"])
        LON_SEKOLAH = float(data_geo["longitude"])
        RADIUS_MAKSIMAL = float(data_geo["radius"])
        print("DATA GEO =", data_geo)
        print("LAT SEKOLAH =", LAT_SEKOLAH)
        print("LON SEKOLAH =", LON_SEKOLAH)
        print("RADIUS =", RADIUS_MAKSIMAL)

        print("LAT CLIENT =", latitude)
        print("LON CLIENT =", longitude)
        distance_geo = hitung_jarak(
            latitude,
            longitude,
            LAT_SEKOLAH,
            LON_SEKOLAH
        )
        
        if distance_geo > RADIUS_MAKSIMAL:
            return JSONResponse(status_code=400, content={
                "status": "gagal",
                "message": f"Gagal: Anda di luar radius sekolah! Jarak Anda: {distance_geo} meter."
            })

        # 4. BUKTI FOTO UNTUK BAB 4 SKRIPSI (OVERLAY BOX & METADATA TEXT)
        if koordinat:
            x, y, w, h = koordinat
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3) # Kotak hijau di wajah
        
        waktu_str = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, f"ID/Name: {id_siswa} - {nama_siswa}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"ArcFace Match: {akurasi}%", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Geo-Dist: {distance_geo} m", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, waktu_str, (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        nama_file_hasil = (
            f"BUKTI_{id_siswa}_{uuid4().hex}.jpg"
        )
        path_foto = os.path.join(
            "hasil_presensi",
            nama_file_hasil
        )

        cv2.imwrite(path_foto, img)
        foto_url = upload_ke_google_drive(path_foto)
        print(foto_url)

        db = get_db_connection()
        cursor = db.cursor()
        try:
            waktu_sekarang = datetime.now(ZoneInfo("Asia/Jakarta"))

            status_kehadiran = (
                "Hadir"
                if waktu_sekarang.time() <= datetime.strptime("08:00:00", "%H:%M:%S").time()
                else "Terlambat"
            )

            sql = """
                INSERT INTO catatan_kehadiran
                (
                    id_siswa,
                    status_kehadiran,
                    distance,
                    jarak_geo,
                    waktu_absen,
                    foto_url
                )
                VALUES (%s,%s,%s,%s,%s,%s)
            """
            cursor.execute(
                sql,
                (
                    int(id_siswa),
                    status_kehadiran,
                    float(akurasi),
                    float(distance_geo),
                    waktu_sekarang,
                    foto_url
                )
            )
            db.commit()

            # Ambil nomor WA orang tua
            cursor.execute("""
                SELECT wa_ortu
                FROM dataset_siswa
                WHERE id_siswa = %s
            """, (int(id_siswa),))

            ortu = cursor.fetchone()

            if ortu and ortu[0]:

                pesan_wa = f"""
            📢 Notifikasi Presensi Siswa

            Nama : {nama_siswa}
            Status : {status_kehadiran}

            Tanggal : {waktu_sekarang.strftime('%d-%m-%Y')}
            Jam : {waktu_sekarang.strftime('%H:%M:%S')} WIB

            SMA Sjakhyakirti Palembang
            """
                try:
                    kirim_wa_fonnte(
                        ortu[0],
                        pesan_wa,
                        path_foto
                    )
                finally:
                    try:
                        os.remove(path_foto)
                        print("FOTO DIHAPUS")
                    except Exception as e:
                        print("GAGAL HAPUS FOTO:", e)

        except Exception as db_err:
            print(f"ERROR SIMPAN PRESENSI: {db_err}")

        finally:
            cursor.close()
            db.close()
            
        return {
            "status": "sukses",
            "nama": nama_siswa,
            "jarak": distance_geo,
            "akurasi": akurasi,
            "pesan": "Presensi Berhasil Terverifikasi Sempurna di Cloud API!"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Terjadi kesalahan internal server: {str(e)}"})

#----BAGIAN ADMIN----
@app.post("/admin/input-siswa")
async def input_siswa_baru(
    nis_siswa: str = Form(...),
    nama_siswa: str = Form(...),
    kelas_siswa: str = Form(...),
    jenis_kelamin: str = Form(...),
    wa_ortu: str = Form(...),
    files: list[UploadFile] = File(...)
):
    try:
        # Validasi jumlah foto
        if len(files) < 5:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "gagal",
                    "message": "Minimal 5 foto diperlukan."
                }
            )
        
        # 1. Konversi file gambar kiriman HP Admin ke OpenCV Frame
        daftar_frame = []
        for file in files:
            img = await upload_to_cv2(file)
            if img is not None:
                img = cv2.resize(img, (1024, 1024))
            if img is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "gagal",
                        "message": f"File {file.filename} tidak valid."
                    }
                )
            daftar_frame.append(img)
         # 2. Membuat centroid embedding
        sukses_ekstrak, vektor_wajah = engine.ekstrak_vektor_centroid(
            daftar_frame
        )   
        if not sukses_ekstrak:
            return JSONResponse(status_code=400, content={"status": "gagal", "message": "Minimal 5 wajah valid diperlukan."})

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
            
            password_default = str(nis_siswa)

            params = (
                str(nis_siswa),
                nama_siswa,
                jenis_kelamin,
                kelas_siswa,
                wa_ortu,
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

 
class AbsenRequest(BaseModel):
    nisn: str
    status: str  


class GeofencingSchema(BaseModel):
    latitude_sekolah: float
    longitude_sekolah: float
    radius_meter: float

class TemplateWASchema(BaseModel):
    template_wa: str     

@app.get("/siswa")
def get_all_siswa():
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Gagal terhubung ke database cloud")
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM dataset_siswa")
        siswa_data = cursor.fetchall()
        return {"status": "sukses", "data": siswa_data}
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/geofencing")
def get_geofencing():
    try:
        data = ambil_pengaturan_geofencing()

        return {
            "latitude_sekolah": data.get("latitude", -2.9602),
            "longitude_sekolah": data.get("longitude", 104.7554),
            "radius_meter": data.get("radius", 50.0)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil konfigurasi geofencing: {str(e)}"
        )

@app.post("/geofencing/update")
def update_geofencing(data: GeofencingSchema):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Gagal terhubung ke database cloud")
        
    cursor = conn.cursor()
    query = """
        INSERT INTO konfigurasi_geofencing (id, latitude, longitude, radius) 
        VALUES (1, %s, %s, %s) 
        ON DUPLICATE KEY UPDATE 
        latitude=%s, longitude=%s, radius=%s
    """
    values = (data.latitude_sekolah, data.longitude_sekolah, data.radius_meter,
              data.latitude_sekolah, data.longitude_sekolah, data.radius_meter)
    try:
        cursor.execute(query, values)
        conn.commit()
        return {"status": "success", "message": "Konfigurasi geofencing berhasil diperbarui di cloud"}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/wa-template")
def get_template_wa():
    data = ambil_pengaturan_geofencing()

    return {
        "template_wa": data.get(
            "template_wa",
            "Presensi [nama] kelas [kelas] tercatat [status] pada [jam]"
        )
    }

@app.post("/wa-template/update")
def update_template_wa(data: TemplateWASchema):

    conn = get_db_connection()

    if not conn:
        raise HTTPException(
            status_code=500,
            detail="Gagal terhubung ke database"
        )

    cursor = conn.cursor()

    try:

        sql = """
        UPDATE konfigurasi_geofencing
        SET template_wa = %s
        WHERE id = 1
        """

        cursor.execute(
            sql,
            (data.template_wa,)
        )

        conn.commit()

        return {
            "status": "success",
            "message": "Template WhatsApp berhasil diperbarui"
        }

    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        cursor.close()
        conn.close()

