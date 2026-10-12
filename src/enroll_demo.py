import glob
import os
import shutil
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
brain_dir = r"C:\Users\SWAPNIL\.gemini\antigravity-ide\brain\3d15f2f1-56fe-4249-9fd5-d4d3eb0159a9"

p1 = glob.glob(os.path.join(brain_dir, "grace_hopper_portrait_*.jpg"))
p2 = glob.glob(os.path.join(brain_dir, "marcus_chen_portrait_*.jpg"))

sample_dir = os.path.join(PROJECT_ROOT, "data", "sample_portraits")
os.makedirs(sample_dir, exist_ok=True)

if p1:
    dest1 = os.path.join(sample_dir, "dr_elena_vance.jpg")
    shutil.copy(p1[0], dest1)
    print("Copied Elena Vance portrait to:", dest1)

if p2:
    dest2 = os.path.join(sample_dir, "marcus_chen.jpg")
    shutil.copy(p2[0], dest2)
    print("Copied Marcus Chen portrait to:", dest2)

import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.database import FaceDatabase

detector = FaceDetector()
recognizer = FaceRecognizer(backend="sface")
db = FaceDatabase()

for name, role, filename in [
    ("Dr. Elena Vance", "Lead AI Scientist", "dr_elena_vance.jpg"),
    ("Marcus Chen", "Principal ML Engineer", "marcus_chen.jpg")
]:
    img_path = os.path.join(sample_dir, filename)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        faces = detector.detect(img, align=True)
        if faces:
            faces.sort(key=lambda f: f.confidence, reverse=True)
            t = faces[0]
            emb = recognizer.extract_embedding(img, raw_face_data=t.raw_face_data)
            x, y, w, h = t.bbox
            pad = int(min(w, h) * 0.15)
            ih, iw = img.shape[:2]
            crop = img[max(0, y - pad):min(ih, y + h + pad), max(0, x - pad):min(iw, x + w + pad)]
            safe_id = name.lower().replace(" ", "_").replace(".", "")
            db.enroll(name=name, embedding=emb, thumbnail_img=crop, role=role, identity_id=safe_id)
            print(f"[✓] Enrolled: {name} ({role})")

print(f"\n[+] Total identities now enrolled: {len(db.identities)}")
