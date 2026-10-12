"""
Facial Recognition and Identification Web Studio Application.
Flask server providing REST APIs, live webcam streaming, and interactive studio UI.
"""

import base64
import os
import sys
import time
from typing import Optional
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response, send_from_directory

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.database import FaceDatabase
from src.stream_processor import StreamProcessor
from src.train import train_model
from src.pose_guide import FacePoseGuide

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize global core services
detector = FaceDetector()
recognizer_sface = FaceRecognizer(backend="sface")
recognizer_pytorch = FaceRecognizer(backend="pytorch")
database = FaceDatabase()
processor = StreamProcessor(detector=detector, recognizer=recognizer_sface, database=database)


def decode_image_from_request(req) -> Optional[np.ndarray]:
    """Helper to decode image from either multipart file or base64 JSON payload."""
    if "image" in req.files:
        file = req.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    elif req.is_json and "image_b64" in req.json:
        b64_str = req.json["image_b64"]
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_str)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    return None


@app.route("/")
def index():
    """Main Web Application Dashboard."""
    return render_template("index.html")


@app.route("/data/thumbnails/<path:filename>")
def get_thumbnail(filename):
    """Serve enrolled face thumbnail crops."""
    thumb_dir = os.path.join(PROJECT_ROOT, "data", "thumbnails")
    return send_from_directory(thumb_dir, filename)


@app.route("/api/identities", methods=["GET"])
def get_identities():
    """Returns all registered identities."""
    return jsonify({
        "status": "success",
        "identities": database.get_all(),
        "count": len(database.identities)
    })


@app.route("/api/enroll", methods=["POST"])
def enroll_identity():
    """Enrolls a new face into the vector gallery."""
    try:
        name = request.form.get("name") or (request.json.get("name") if request.is_json else None)
        role = request.form.get("role") or (request.json.get("role") if request.is_json else "Member")
        backend = request.form.get("backend") or (request.json.get("backend") if request.is_json else "sface")

        if not name or not name.strip():
            return jsonify({"status": "error", "message": "Name is required."}), 400

        img = decode_image_from_request(request)
        if img is None:
            return jsonify({"status": "error", "message": "Valid face image is required."}), 400

        faces = detector.detect(img, align=True)
        if not faces:
            return jsonify({"status": "error", "message": "No face detected in the image. Please use a clear, front-facing portrait."}), 400

        faces.sort(key=lambda f: f.confidence, reverse=True)
        target = faces[0]

        rec = recognizer_pytorch if backend == "pytorch" else recognizer_sface
        embedding = rec.extract_embedding(img, raw_face_data=target.raw_face_data)

        # Crop thumbnail
        x, y, w, h = target.bbox
        pad = int(min(w, h) * 0.15)
        ih, iw = img.shape[:2]
        crop = img[max(0, y - pad):min(ih, y + h + pad), max(0, x - pad):min(iw, x + w + pad)]

        identity_id = database.enroll(
            name=name.strip(),
            embedding=embedding,
            thumbnail_img=crop,
            role=role.strip() if role else "Member"
        )

        return jsonify({
            "status": "success",
            "message": f"Successfully enrolled {name}",
            "identity_id": identity_id,
            "confidence": round(target.confidence * 100, 1),
            "total_identities": len(database.identities)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analyze_pose", methods=["POST"])
def analyze_pose():
    """Analyzes a webcam frame and provides real-time guidance on face pose and alignment."""
    try:
        target_pose = request.json.get("target_pose", "center") if request.is_json else "center"
        img = decode_image_from_request(request)
        if img is None:
            return jsonify({"status": "error", "message": "Valid frame required"}), 400

        h, w = img.shape[:2]
        faces = detector.detect(img, align=False)
        target_face = faces[0] if faces else None

        result = FacePoseGuide.evaluate(target_face, frame_width=w, frame_height=h, target_pose=target_pose)
        
        return jsonify({
            "status": "success",
            "face_detected": result.face_detected,
            "pose_matched": result.pose_matched,
            "target_pose": result.target_pose,
            "estimated_pose": result.estimated_pose,
            "instruction": result.instruction,
            "yaw_ratio": result.yaw_ratio,
            "pitch_ratio": result.pitch_ratio,
            "distance_status": result.distance_status,
            "is_centered": result.is_centered,
            "quality_score": result.quality_score,
            "quality_ok": result.quality_ok,
            "target_oval": result.target_oval,
            "bbox": target_face.bbox if target_face else None,
            "landmarks": target_face.landmarks if target_face else []
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/enroll_multisample", methods=["POST"])
def enroll_multisample():
    """Enrolls an identity using multiple captured sample images (e.g. Center, Left, Right, Up, Smile)."""
    try:
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        role = data.get("role", "Member").strip()
        backend = data.get("backend", "sface")
        samples_b64 = data.get("samples", [])
        labels = data.get("labels", [])

        if not name:
            return jsonify({"status": "error", "message": "Name is required."}), 400
        if not samples_b64:
            return jsonify({"status": "error", "message": "At least one sample image is required."}), 400

        rec = recognizer_pytorch if backend == "pytorch" else recognizer_sface

        embeddings = []
        thumbnail_crops = []

        for idx, s_b64 in enumerate(samples_b64):
            if "," in s_b64:
                s_b64 = s_b64.split(",", 1)[1]
            raw_bytes = base64.b64decode(s_b64)
            img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            faces = detector.detect(img, align=True)
            if not faces:
                continue

            faces.sort(key=lambda f: f.confidence, reverse=True)
            target = faces[0]

            emb = rec.extract_embedding(img, raw_face_data=target.raw_face_data)
            embeddings.append(emb)

            x, y, w, h = target.bbox
            pad = int(min(w, h) * 0.15)
            ih, iw = img.shape[:2]
            crop = img[max(0, y - pad):min(ih, y + h + pad), max(0, x - pad):min(iw, x + w + pad)]
            thumbnail_crops.append(crop)

        if not embeddings:
            return jsonify({"status": "error", "message": "Could not detect valid faces in provided sample frames."}), 400

        identity_id = database.enroll_multisample(
            name=name,
            embeddings=embeddings,
            thumbnail_imgs=thumbnail_crops,
            role=role if role else "Member"
        )

        return jsonify({
            "status": "success",
            "message": f"Successfully enrolled {name} with {len(embeddings)} facial feature samples",
            "identity_id": identity_id,
            "sample_count": len(embeddings),
            "total_identities": len(database.identities)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/identity/<identity_id>", methods=["DELETE"])
def delete_identity(identity_id):
    """Deletes an enrolled identity."""
    success = database.delete_identity(identity_id)
    if success:
        return jsonify({"status": "success", "message": f"Deleted identity {identity_id}"})
    return jsonify({"status": "error", "message": "Identity not found."}), 404


@app.route("/api/identify", methods=["POST"])
def identify_faces():
    """Identifies all faces detected in an image against the vector gallery."""
    try:
        threshold = float(request.form.get("threshold", 0.363)) if request.form else 0.363
        if request.is_json and "threshold" in request.json:
            threshold = float(request.json["threshold"])

        backend = request.form.get("backend", "sface") if request.form else "sface"
        if request.is_json and "backend" in request.json:
            backend = request.json["backend"]

        img = decode_image_from_request(request)
        if img is None:
            return jsonify({"status": "error", "message": "Valid image is required."}), 400

        faces = detector.detect(img, align=True)
        rec = recognizer_pytorch if backend == "pytorch" else recognizer_sface

        results = []
        annotated = img.copy()

        for idx, face in enumerate(faces):
            emb = rec.extract_embedding(img, raw_face_data=face.raw_face_data)
            matches = database.identify(emb, threshold=threshold, top_k=3)
            best = matches[0]

            x, y, w, h = face.bbox
            color = (74, 222, 128) if best.is_identified else (80, 100, 248) # BGR
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"{best.name} ({best.confidence:.0f}%)" if best.is_identified else "Unknown"
            cv2.putText(annotated, label, (x, max(24, y - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)

            results.append({
                "face_index": idx + 1,
                "bbox": [x, y, w, h],
                "confidence": best.confidence,
                "is_identified": best.is_identified,
                "name": best.name,
                "role": best.role or "Unregistered",
                "identity_id": best.identity_id,
                "cosine_score": round(best.cosine_similarity, 4),
                "euclidean_distance": round(best.euclidean_distance, 4),
                "top_candidates": [
                    {
                        "name": m.name,
                        "role": m.role,
                        "confidence": m.confidence,
                        "cosine": round(m.cosine_similarity, 4)
                    } for m in matches
                ],
                "vector_preview": [round(float(v), 3) for v in emb[:8]] # first 8 dims for display
            })

        # Encode annotated image to base64
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "status": "success",
            "face_count": len(faces),
            "faces": results,
            "annotated_image": annotated_b64
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/verify", methods=["POST"])
def verify_faces():
    """1:1 Face Verification between two uploaded images."""
    try:
        threshold = 0.363
        backend = "sface"

        if "threshold" in request.form:
            threshold = float(request.form["threshold"])
        elif request.is_json and "threshold" in request.json:
            threshold = float(request.json["threshold"])

        if "backend" in request.form:
            backend = request.form["backend"]
        elif request.is_json and "backend" in request.json:
            backend = request.json["backend"]

        # Image 1 & Image 2
        img1, img2 = None, None
        if "image1" in request.files and "image2" in request.files:
            b1 = np.frombuffer(request.files["image1"].read(), np.uint8)
            b2 = np.frombuffer(request.files["image2"].read(), np.uint8)
            img1 = cv2.imdecode(b1, cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(b2, cv2.IMREAD_COLOR)
        elif request.is_json and "image1_b64" in request.json and "image2_b64" in request.json:
            b64_1 = request.json["image1_b64"].split(",", 1)[-1]
            b64_2 = request.json["image2_b64"].split(",", 1)[-1]
            img1 = cv2.imdecode(np.frombuffer(base64.b64decode(b64_1), np.uint8), cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(np.frombuffer(base64.b64decode(b64_2), np.uint8), cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            return jsonify({"status": "error", "message": "Two valid images are required for verification."}), 400

        faces1 = detector.detect(img1, align=True)
        faces2 = detector.detect(img2, align=True)

        if not faces1:
            return jsonify({"status": "error", "message": "No face detected in Image 1."}), 400
        if not faces2:
            return jsonify({"status": "error", "message": "No face detected in Image 2."}), 400

        rec = recognizer_pytorch if backend == "pytorch" else recognizer_sface
        emb1 = rec.extract_embedding(img1, raw_face_data=faces1[0].raw_face_data)
        emb2 = rec.extract_embedding(img2, raw_face_data=faces2[0].raw_face_data)

        is_match, cos_score, l2_dist = rec.is_match(emb1, emb2, custom_threshold=threshold)

        # Confidence percentage mapping
        if cos_score >= threshold:
            conf = 60.0 + (cos_score - threshold) / max(0.01, (1.0 - threshold)) * 40.0
        else:
            conf = max(0.0, (cos_score / max(0.01, threshold)) * 60.0)

        return jsonify({
            "status": "success",
            "is_match": bool(is_match),
            "verdict": "MATCH: SAME IDENTITY" if is_match else "NO MATCH: DIFFERENT PERSONS",
            "confidence": round(conf, 1),
            "cosine_similarity": round(float(cos_score), 4),
            "euclidean_distance": round(float(l2_dist), 4),
            "threshold": threshold,
            "vector1_sample": [round(float(v), 3) for v in emb1[:8]],
            "vector2_sample": [round(float(v), 3) for v in emb2[:8]]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stream_frame", methods=["POST"])
def stream_frame():
    """Processes a single webcam snapshot frame and returns HUD annotated image + telemetry."""
    try:
        threshold = 0.363
        if request.is_json and "threshold" in request.json:
            threshold = float(request.json["threshold"])

        include_annotated = True
        if request.is_json and "include_annotated" in request.json:
            include_annotated = bool(request.json["include_annotated"])

        img = decode_image_from_request(request)
        if img is None:
            return jsonify({"status": "error", "message": "Invalid frame data"}), 400

        t0 = time.time()
        annotated, results = processor.process_frame(img, draw_annotations=include_annotated, threshold=threshold)
        proc_ms = round((time.time() - t0) * 1000.0, 1)

        annotated_b64 = None
        if include_annotated:
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "status": "success",
            "latency_ms": proc_ms,
            "detections": results,
            "annotated_frame": annotated_b64
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    """Runs pipeline benchmark on the host system."""
    try:
        runs = 20
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.ellipse(test_img, (320, 240), (100, 130), 0, 0, 360, (210, 190, 170), -1)

        # 1. Detection
        t0 = time.time()
        for _ in range(runs):
            detector.detect(test_img)
        det_ms = (time.time() - t0) / runs * 1000.0

        # 2. SFace ONNX
        mock_face = np.array([220, 110, 200, 260, 280, 200, 360, 200, 320, 250, 290, 310, 350, 310, 0.99], dtype=np.float32)
        t0 = time.time()
        for _ in range(runs):
            recognizer_sface.extract_embedding(test_img, raw_face_data=mock_face)
        sface_ms = (time.time() - t0) / runs * 1000.0

        # 3. PyTorch MobileFaceNet
        crop_112 = np.zeros((112, 112, 3), dtype=np.uint8)
        t0 = time.time()
        for _ in range(runs):
            recognizer_pytorch.extract_embedding(crop_112)
        pt_ms = (time.time() - t0) / runs * 1000.0

        # 4. 1:N Search
        test_emb = np.random.randn(128).astype(np.float32)
        t0 = time.time()
        for _ in range(runs * 5):
            database.identify(test_emb)
        db_ms = (time.time() - t0) / (runs * 5) * 1000.0

        total_ms = det_ms + sface_ms + db_ms

        return jsonify({
            "status": "success",
            "detection_ms": round(det_ms, 2),
            "detection_fps": round(1000.0 / det_ms, 1),
            "sface_ms": round(sface_ms, 2),
            "sface_fps": round(1000.0 / sface_ms, 1),
            "pytorch_ms": round(pt_ms, 2),
            "pytorch_fps": round(1000.0 / pt_ms, 1),
            "db_search_ms": round(db_ms, 4),
            "pipeline_ms": round(total_ms, 2),
            "pipeline_fps": round(1000.0 / total_ms, 1)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Returns recent recognition event logs from attendance.csv."""
    log_file = processor.log_file
    if not os.path.exists(log_file):
        return jsonify({"status": "success", "logs": []})

    entries = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Skip header, get last 50 reversed
        for line in reversed(lines[1:]):
            parts = [p.strip().strip('"') for p in line.strip().split(",")]
            if len(parts) >= 6:
                entries.append({
                    "timestamp": parts[0],
                    "identity_id": parts[1],
                    "name": parts[2],
                    "role": parts[3],
                    "confidence": parts[4],
                    "cosine_score": parts[5]
                })
    except Exception:
        pass

    return jsonify({"status": "success", "logs": entries[:50]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  FACIAL RECOGNITION & IDENTIFICATION STUDIO")
    print(f"  Server starting at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
