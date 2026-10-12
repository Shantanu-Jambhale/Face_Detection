import os
import urllib.request
import sys

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

MODELS = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "expected_min_bytes": 200_000,
        "description": "YuNet Face Detector (5 landmarks + bounding box)"
    },
    "sface": {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "expected_min_bytes": 35_000_000,
        "description": "SFace Deep Neural Network (128-D Feature Embedding)"
    }
}

def download_file(url: str, dest_path: str, description: str):
    print(f"[*] Downloading {description}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 64
        
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded / total_size * 100
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r    Progress: {percent:5.1f}% ({mb_down:5.1f} / {mb_total:5.1f} MB)")
                sys.stdout.flush()
    print(f"\n[+] Successfully saved to: {dest_path}")

def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    paths = {}
    for key, info in MODELS.items():
        dest = os.path.join(MODEL_DIR, info["filename"])
        paths[key] = dest
        if not os.path.exists(dest) or os.path.getsize(dest) < info["expected_min_bytes"]:
            download_file(info["url"], dest, info["description"])
        else:
            print(f"[✓] Model {info['filename']} already downloaded ({os.path.getsize(dest):,} bytes).")
    return paths

if __name__ == "__main__":
    ensure_models()
