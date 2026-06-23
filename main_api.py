import numpy as np
import os
import json
import cv2
from fastapi import Request
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
from datetime import datetime
from absen_engine import AbsenEngine, get_db_connection
from pydantic import BaseModel
from mysql.connector import Error
from zoneinfo import ZoneInfo
from uuid import uuid4
import requests
from fastapi.responses import FileResponse

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

@app.get("/foto-presensi/{nama_file}")
def lihat_foto_presensi(nama_file: str):

    path = os.path.join(
        "hasil_presensi",
        nama_file
    )

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Foto tidak ditemukan"
        )

    return FileResponse(path)

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
        foto_url = (
            f"https://sjakhyakirtibackendapi-production.up.railway.app/foto-presensi/{nama_file_hasil}"
        )
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
                SELECT wa_ortu, kelas
                FROM dataset_siswa
                WHERE id_siswa = %s
            """, (int(id_siswa),))

            ortu = cursor.fetchone()

            if ortu and ortu[0]:

                    nomor_wa = ortu[0]
                    kelas_siswa = ortu[1]

                    template_wa = data_geo.get(
                        "template_wa",
                        "Presensi [nama] kelas [kelas] tercatat [status] pada [jam]"
                    )

                    pesan_wa = (
                        template_wa
                        .replace("{nama_siswa}", nama_siswa)
                        .replace("{kelas}", kelas_siswa)
                        .replace("{status_kehadiran}", status_kehadiran)
                        .replace("{tanggal}", waktu_sekarang.strftime("%d-%m-%Y"))
                        .replace("{jam}", waktu_sekarang.strftime("%H:%M:%S"))
                        .replace("{foto_url}", foto_url)
                    )

                    status_wa = kirim_wa_fonnte(
                        nomor_wa,
                        pesan_wa,
                        path_foto
                    )
                   
        except Exception as db_err:
            print(f"ERROR SIMPAN PRESENSI: {db_err}")

        finally:
            cursor.close()
            db.close()
            
        return {
            "status": "sukses",
            "nama": nama_siswa,
            "status_kehadiran": status_kehadiran,
            "status_wa": status_wa,
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

@app.get("/gps-test")
async def gps_test():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GPS Test</title>
    </head>

    <body>
        <h2>GPS Test SMA Sjakhyakirti</h2>

        <button onclick="ambilGPS()">
            Ambil Lokasi Saya
        </button>

        <p id="hasil">
            Belum ada lokasi
        </p>

        <script>
        function ambilGPS() {

            navigator.geolocation.getCurrentPosition(

                function(pos) {

                    document.getElementById("hasil").innerHTML =
                        "Latitude: " +
                        pos.coords.latitude +
                        "<br>Longitude: " +
                        pos.coords.longitude;

                    fetch("/cek-lokasi", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            latitude: pos.coords.latitude,
                            longitude: pos.coords.longitude
                        })
                    })
                    .then(response => {
                        console.log("STATUS =", response.status);
                        return response.text();
                    })
                    .then(data => {
                        console.log("RESP =", data);

                        document.getElementById("hasil").innerHTML =
                            "RESPON SERVER:<br>" + data;
                    })
                    .catch(err => {
                        document.getElementById("hasil").innerHTML =
                            "ERROR FETCH:<br>" + err;
                    });
                },

                function(err) {

                    alert(
                        "Gagal mengambil GPS: " +
                        err.message
                    );

                }

            );

        }
        </script>

    </body>
    </html>
    """)

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

class CekLokasiSchema(BaseModel):
    latitude: float
    longitude: float

@app.get("/cek-lokasi-test")
def cek_lokasi_test():
    return {
        "status": "aktif"
    }

@app.post("/cek-lokasi")
def cek_lokasi(data: CekLokasiSchema):

    print("MASUK CEK LOKASI")
    print("LAT =", data.latitude)
    print("LON =", data.longitude)

    data_geo = ambil_pengaturan_geofencing()

    jarak = hitung_jarak(
        data.latitude,
        data.longitude,
        float(data_geo["latitude"]),
        float(data_geo["longitude"])
    )

    jarak = float(jarak)

    geo_ok = jarak <= float(data_geo["radius"])

    return {
        "geo_ok": bool(geo_ok),
        "jarak": round(jarak, 2),
        "radius": float(data_geo["radius"])
    }

@app.get("/presensi-web")
async def presensi_web(
    id_siswa: int,
    nama: str
):
    return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>

    <title>Presensi SMA Sjakhyakirti</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    body {{

        font-family: 'Poppins', sans-serif;

        min-height: 100vh;

        background: linear-gradient(
            180deg,
            #2563EB 0%,
            #60A5FA 100%
        );

        padding: 20px;

        display: flex;
        justify-content: center;
        align-items: center;
    }}

    .card {{

        width: 100%;
        max-width: 520px;

        background: #FFFFFF;

        border-radius: 24px;

        padding: 28px;

        box-shadow:
            0 15px 40px rgba(0,0,0,0.15);

        animation: fadeIn .4s ease;
    }}

    @keyframes fadeIn {{

        from {{
            opacity: 0;
            transform: translateY(20px);
        }}

        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    h2 {{

        text-align: center;

        color: #0F172A;

        font-size: 26px;

        font-weight: 700;

        margin-bottom: 20px;
    }}

    p {{
        color: #334155;
        margin-bottom: 12px;
    }}

    button {{

        width: 100%;

        padding: 15px;

        border: none;

        border-radius: 14px;

        background: #2563EB;

        color: white;

        font-size: 15px;

        font-weight: 600;

        cursor: pointer;

        transition: .2s;
    }}

    button:hover {{
        background: #1D4ED8;
    }}

    button:active {{
        transform: scale(.98);
    }}

    #hasil {{

        margin-top: 20px;

        text-align: center;

        font-size: 15px;

        color: #0F172A;
    }}

    input[type=file] {{

        width: 100%;

        margin-top: 12px;

        padding: 12px;

        border: 2px dashed #93C5FD;

        border-radius: 14px;

        background: #EFF6FF;
    }}

    pre {{

        text-align: left;

        white-space: pre-wrap;

        word-break: break-word;

        background: #F8FAFC;

        border: 1px solid #E2E8F0;

        padding: 14px;

        border-radius: 14px;

        font-size: 13px;
    }}

    .success-box {{

        background: #ECFDF5;

        border: 1px solid #A7F3D0;

        border-radius: 16px;

        padding: 16px;
    }}

    .error-box {{

        background: #FEF2F2;

        border: 1px solid #FECACA;

        border-radius: 16px;

        padding: 16px;
    }}

    .geo-box {{

        background: #EFF6FF;

        border: 1px solid #BFDBFE;

        border-radius: 16px;

        padding: 16px;
    }}

    .loading {{

        color: #2563EB;

        font-weight: 600;

        text-align: center;
    }}

    .loading {{

        color: #2563EB;

        font-weight: 600;

        text-align: center;
        }}

    .info-card {{

        background:#F8FAFC;

        border:1px solid #E2E8F0;

        border-radius:18px;

        padding:18px;

        margin-bottom:20px;
        }}

    .info-row {{

        display:flex;

        justify-content:space-between;

        align-items:center;

        margin-bottom:12px;
       }}

        .info-row:last-child {{

        margin-bottom:0;
        }}

        .label {{

        color:#64748B;

        font-size:14px;
        }}

        .value {{

        color:#0F172A;

        font-weight:600;
        }}

        .badge {{

        background:#DBEAFE;

        color:#1D4ED8;

        padding:5px 12px;

        border-radius:999px;

        font-size:12px;

        font-weight:700;
        }}
    </style>
        
</head>

<body>

<div class="card">

    <div class="header">

        <div class="icon">
            📍
        </div>

        <h2>
            Presensi SMA Sjakhyakirti
        </h2>

        <p>
            Sistem Presensi Berbasis Pengenalan Wajah dan Geofencing
        </p>

    </div>

    <div class="info-card">

        <div class="info-row">

            <span class="label">
                ID Siswa
            </span>

            <span class="badge">
                {id_siswa}
            </span>

        </div>

        <div class="info-row">

            <span class="label">
                Nama
            </span>

            <span class="value">
                {nama}
            </span>

        </div>

    </div>

    <input
        type="hidden"
        id="id_siswa"
        value="{id_siswa}">

    <input
        type="hidden"
        id="nama_siswa"
        value="{nama}">

    <button onclick="ambilGPS()">
        Ambil Lokasi Saya
    </button>

    <div id="hasil">
        Menunggu GPS...
    </div>

</div>

<script>

async function uploadPresensi() {{

    const fileInput =
        document.getElementById("foto");

    if (!fileInput.files.length) {{

        alert(
            "Silakan ambil selfie terlebih dahulu."
        );

        return;
    }}

    let formData = new FormData();

    formData.append(
        "id_siswa",
        document.getElementById("id_siswa").value
    );

    formData.append(
        "nama_siswa",
        document.getElementById("nama_siswa").value
    );

    formData.append(
        "latitude",
        window.latGPS
    );

    formData.append(
        "longitude",
        window.lonGPS
    );

    formData.append(
        "file",
        fileInput.files[0]
    );

    document.getElementById("hasil").innerHTML = `

    <div class="info-card">

        <div class="loading">

            ⏳ Memverifikasi wajah...

            <br><br>

            Mohon tunggu beberapa detik

        </div>

    </div>

    `;
    try {{

        const response = await fetch(
            "/verify-presensi",
            {{
                method: "POST",
                body: formData
            }}
        );

        const data =
            await response.json();

        document.getElementById("hasil").innerHTML =
            "<pre>" +
            JSON.stringify(
                data,
                null,
                2
            ) +
            "</pre>";

    }}
    catch(err) {{

        document.getElementById("hasil").innerHTML =
            "ERROR: " + err;

    }}

}}

function ambilGPS() {{

    navigator.geolocation.getCurrentPosition(

        function(pos) {{

            window.latGPS =
                pos.coords.latitude;

            window.lonGPS =
                pos.coords.longitude;

            document.getElementById("hasil").innerHTML =
                "Memeriksa geofencing...";

            fetch("/cek-lokasi", {{

                method: "POST",

                headers: {{
                    "Content-Type":
                    "application/json"
                }},

                body: JSON.stringify({{
                    latitude:
                        pos.coords.latitude,
                    longitude:
                        pos.coords.longitude
                }})

            }})

            .then(response =>
                response.json()
            )

            .then(data => {{

            let html = "";

            if (data.geo_ok) {{

                html = `

                <div class="geo-box">

                    <div style="
                        color:#16A34A;
                        font-weight:700;
                        font-size:18px;
                        margin-bottom:12px;
                    ">
                        ✅ Dalam Radius Sekolah
                    </div>

                    <div style="
                        color:#475569;
                        margin-bottom:16px;
                    ">
                        Lokasi Anda berhasil diverifikasi.
                    </div>

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-bottom:10px;
                    ">
                        <span>Jarak Anda</span>
                        <b>${{data.jarak}} meter</b>
                    </div>

                    <div style="
                        display:flex;
                        justify-content:space-between;
                    ">
                        <span>Radius Sekolah</span>
                        <b>${{data.radius}} meter</b>
                    </div>

                </div>

                <div style="height:20px"></div>

                <div class="geo-box">

                    <div style="
                        font-size:18px;
                        font-weight:700;
                        color:#0F172A;
                        margin-bottom:10px;
                    ">
                        📸 Ambil Selfie
                    </div>

                    <div style="
                        color:#64748B;
                        font-size:14px;
                        margin-bottom:15px;
                    ">
                        Pastikan wajah terlihat jelas dan pencahayaan cukup.
                    </div>

                    <input
                        type="file"
                        id="foto"
                        accept="image/*"
                        capture="user">

                    <br><br>

                    <button onclick="uploadPresensi()">
                        🚀 Kirim Presensi
                    </button>

                </div>

                `;

            }}
            else {{

                html = `

                <div class="error-box">

                    <div style="
                        color:#DC2626;
                        font-weight:700;
                        font-size:18px;
                        margin-bottom:12px;
                    ">
                        ❌ Di Luar Radius Sekolah
                    </div>

                    <div style="
                        color:#7F1D1D;
                        margin-bottom:16px;
                    ">
                        Anda berada di luar area presensi yang diizinkan.
                    </div>

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-bottom:10px;
                    ">
                        <span>Jarak Anda</span>
                        <b>${{data.jarak}} meter</b>
                    </div>

                    <div style="
                        display:flex;
                        justify-content:space-between;
                    ">
                        <span>Radius Maksimal</span>
                        <b>${{data.radius}} meter</b>
                    </div>

                </div>

                `;

            }}

            document.getElementById("hasil").innerHTML = html;


            }})

            .catch(err => {{

                document.getElementById(
                    "hasil"
                ).innerHTML =
                    "ERROR: " + err;

            }});

        }},

        function(err) {{

            alert(
                "Gagal mengambil GPS: " +
                err.message
            );

        }}

    );

}}

</script>

</body>
</html>
""")