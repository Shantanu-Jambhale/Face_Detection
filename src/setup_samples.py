"""
Populates sample portraits and enrolls initial pioneer identities
into the vector database for immediate out-of-the-box testing.
"""

import os
import sys
import urllib.request
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.database import FaceDatabase

SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "sample_portraits")
os.makedirs(SAMPLE_DIR, exist_ok=True)

SAMPLES = [
    {
        "name": "Alan Turing",
        "role": "Computer Pioneer & Cryptanalyst",
        "filename": "alan_turing.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a1/Alan_Turing_Aged_16.jpg"
    },
    {
        "name": "Grace Hopper",
        "role": "Computer Scientist & US Navy Rear Admiral",
        "filename": "grace_hopper.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/ad/Commodore_Grace_M._Hopper%2C_USN_%28covered%29.jpg"
    },
    {
        "name": "Ada Lovelace",
        "role": "Mathematician & First Computer Programmer",
        "filename": "ada_lovelace.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a4/Ada_Lovelace_portrait.jpg"
    }
]

def setup_samples():
    detector = FaceDetector()
    recognizer = FaceRecognizer(backend="sface")
    db = FaceDatabase()

    print("[*] Setting up sample pioneer identities...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for item in SAMPLES:
        dest_path = os.path.join(SAMPLE_DIR, item["filename"])
        if not os.path.exists(dest_path):
            try:
                print(f"    Downloading portrait for {item['name']}...")
                req = urllib.request.Request(item["url"], headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp, open(dest_path, "wb") as f:
                    f.write(resp.read())
            except Exception as e:
                print(f"    [!] Failed to download {item['name']}: {e}")
                continue

        img = cv2.imread(dest_path)
        if img is None:
            continue

        faces = detector.detect(img, align=True)
        if not faces:
            print(f"    [!] No face detected in {dest_path}")
            continue

        target = faces[0]
        emb = recognizer.extract_embedding(img, raw_face_data=target.raw_face_data)

        # Crop thumbnail
        x, y, w, h = target.bbox
        pad = int(min(w, h) * 0.15)
        ih, iw = img.shape[:2]
        crop = img[max(0, y - pad):min(ih, y + h + pad), max(0, x - pad):min(iw, x + w + pad)]

        safe_id = item["name"].lower().replace(" ", "_")
        db.enroll(name=item["name"], embedding=emb, thumbnail_img=crop, role=item["role"], identity_id=safe_id)
        print(f"    [✓] Enrolled: {item['name']} ({item['role']})")

    print(f"\n[+] Total registered identities in database: {len(db.identities)}")

if __name__ == "__main__":
    setup_samples()
