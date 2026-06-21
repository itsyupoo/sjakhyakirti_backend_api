import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
import cv2
import numpy as np
import json
from deepface import DeepFace
from sklearn.preprocessing import normalize
import mysql.connector
from datetime import datetime

MODEL_NAME = "ArcFace"
DETECTOR = "mtcnn"
BEST_THRESHOLD = 0.35  

def get_db_connection():
    return mysql.connector.connect(
        host="acela.proxy.rlwy.net",
        user="root",
        password="WvjsVeVkcyvgqYyLIEiHBTRgLFkzWAzK",
        database="railway",
        port=33414
    )

class AbsenEngine:
    def __init__(self):
        self.known_ids = []
        self.known_names = []
        self.known_embeddings = []
        self.load_database_wajah()

    def cek_sudah_absen_hari_ini(self, id_siswa):
        db = get_db_connection()
        if not db:
            return False
        cursor = db.cursor()
        hari_ini = datetime.now().strftime('%Y-%m-%d')
        try:
            sql = "SELECT id_siswa FROM catatan_kehadiran WHERE id_siswa = %s AND DATE(waktu_absen) = %s"
            params = (int(id_siswa), str(hari_ini))
            cursor.execute(sql, params)
            result = cursor.fetchone()
            return True if result else False
        except Exception as e:
            print(f"❌ API ENGINE: Gagal validasi harian: {e}")
            return False
        finally:
            cursor.close()
            db.close()

    def load_database_wajah(self):
        """Memuat embedding centroid langsung dari database MySQL Railway"""
        db = get_db_connection()
        cursor = db.cursor()
        self.known_ids = []
        self.known_names = []
        self.known_embeddings = []
        
        try:
            cursor.execute("SELECT id_siswa, nama, face_embedding FROM dataset_siswa WHERE role = 'siswa' AND face_embedding IS NOT NULL")
            rows = cursor.fetchall()
            
            for row in rows:
                id_s = row[0]
                nama = row[1]
                embedding_raw = row[2]
            
                try:
                    if isinstance(embedding_raw, str):
                        embedding_list = json.loads(embedding_raw)
                    else:
                        embedding_list = embedding_raw
                    
                    emb_np = np.array(embedding_list).astype('float32')
                    
                    if emb_np.shape == (512,):
                        self.known_ids.append(id_s)
                        self.known_names.append(nama)
                        self.known_embeddings.append(emb_np)
                except Exception as e:
                    print(f"Embedding error: {e}")
            
            if len(self.known_embeddings) > 0:
                self.known_embeddings = np.array(self.known_embeddings).astype('float32')
                if len(self.known_embeddings.shape) == 3:
                    self.known_embeddings = np.squeeze(self.known_embeddings, axis=1)
                self.known_embeddings = normalize(self.known_embeddings)
                print(f"✅ ENGINE: Berhasil sinkronisasi {len(self.known_names)} wajah dari Database Railway.")
        except Exception as e:
            print(f"❌ ENGINE DATABASE ERROR: {e}")
        finally:
            cursor.close()
            db.close()

    def verifikasi_wajah_api(self, frame, target_id):
        """Fungsi utama untuk memproses wajah kiriman dari API Android"""
        # Cek limit harian lewat database
        if self.cek_sudah_absen_hari_ini(target_id):
            return False, "SUDAH_ABSEN", 0, None

        try:
            print("=== SEBELUM DEEPFACE ===")
            results = DeepFace.represent(
                img_path = frame,
                model_name = MODEL_NAME,
                detector_backend = DETECTOR,
                enforce_detection = True,
                align = True
            )
            print("=== SESUDAH DEEPFACE ===")
            
            for res in results:
                obj = res["facial_area"]
                x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
                
                test_emb = np.array(res["embedding"]).reshape(1, -1)
                test_emb = normalize(test_emb)
                
                if len(self.known_embeddings) > 0:
                    similarity = np.dot(test_emb, self.known_embeddings.T)[0]
                    best_idx = np.argmax(similarity)
                    distance = float(1 - similarity[best_idx])
                    
                    id_terdeteksi = self.known_ids[best_idx]
                    nama_db = self.known_names[best_idx]
                    
                    # Logika Threshold ArcFace
                    if distance <= BEST_THRESHOLD:
                        if str(id_terdeteksi) == str(target_id):
                            # WAJAH COCOK DAN SESUAI AKUN LOGIN
                            akurasi_persen = round((1 - distance) * 100, 2)
                            return True, nama_db, akurasi_persen, (x, y, w, h)
                        else:
                            return False, "BUKAN_PEMILIK_AKUN", round((1 - distance) * 100, 2), (x, y, w, h)
                    else:
                        return False, "WAJAH_TIDAK_COCOK", round((1 - distance) * 100, 2), (x, y, w, h)

            return False, "TIDAK_DIKENALI", 0, None
        except Exception as e:
            # Wajah tidak terdeteksi oleh MTCNN
            return False, "WAJAH_TIDAK_TERDETEKSI", 0, None
        
    def ekstrak_vektor_master(self, frame):
        """
        Fungsi khusus Admin untuk mendeteksi wajah via MTCNN dan mengekstrak 
        vektor wajah (512 dimensi) via ArcFace menggunakan DeepFace.
        """
        try:
            # Menggunakan DeepFace.represent yang sama dengan sistem absen siswa
            results = DeepFace.represent(
                img_path = frame,
                model_name = MODEL_NAME,
                detector_backend = DETECTOR,
                enforce_detection = True,
                align = True
            )

            # Jika wajah sukses terdeteksi, ambil hasil dari wajah pertama
            if len(results) > 0:
                res = results[0]
                
                # Ambil data embedding asli dari ArcFace (berupa list 512 angka)
                vektor_wajah = res["embedding"]
                
                # Mengembalikan status Sukses (True) dan data vektornya
                return True, vektor_wajah

            return False, None

        except Exception as e:
            # Jika MTCNN gagal mendeteksi wajah dalam foto master
            print(f"❌ API ENGINE: Gagal ekstraksi wajah master: {e}")
            return False, None
        
    def ekstrak_vektor_centroid(self, daftar_frame):
        """
        Menerima banyak frame (minimal 5), menghasilkan 1 centroid embedding.
        """
        if len(daftar_frame) < 5:
            return False, None

        all_embeddings = []

        try:
            for frame in daftar_frame:

                results = DeepFace.represent(
                    img_path=frame,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR,
                    enforce_detection=True,
                    align=True
                )

                if len(results) == 0:
                    continue

                # Ambil embedding wajah
                embedding = results[0]["embedding"]

                # Simpan ke list untuk dihitung centroid
                all_embeddings.append(embedding)

            # Minimal 5 wajah berhasil diekstrak
            if len(all_embeddings) < 5:
                return False, None

            centroid = np.mean(
                np.array(all_embeddings),
                axis=0
            )

            return True, centroid.tolist()

        except Exception as e:
            print(f"❌ API ENGINE: Gagal membuat centroid: {e}")
            return False, None
        
        except Exception as e:
            print(f"❌ API ENGINE: Gagal membuat centroid: {e}")
            return False, None    