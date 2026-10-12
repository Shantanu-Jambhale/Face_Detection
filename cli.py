"""
Facial Recognition and Identification Command Line Interface.
Supports enrollment, 1:1 verification, 1:N identification, live camera feeds,
performance benchmarking, and model training.
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

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


def cmd_enroll(args):
    """Enrolls an individual's face into the identity database."""
    img_path = args.image
    name = args.name.strip()
    role = args.role.strip() if args.role else "Member"

    if not os.path.exists(img_path):
        print(f"[!] Error: Image file not found at '{img_path}'.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Error: Could not decode image at '{img_path}'.")
        return

    detector = FaceDetector()
    recognizer = FaceRecognizer(backend=args.backend)
    db = FaceDatabase()

    print(f"[*] Detecting face in '{img_path}'...")
    faces = detector.detect(img, align=True)
    if not faces:
        print(f"[!] No face detected in '{img_path}'. Please use a clearer, front-facing portrait.")
        return

    # Use the highest confidence face if multiple are detected
    faces.sort(key=lambda f: f.confidence, reverse=True)
    target_face = faces[0]
    print(f"[+] Detected face with {target_face.confidence * 100:.1f}% confidence.")

    # Extract embedding
    embedding = recognizer.extract_embedding(img, raw_face_data=target_face.raw_face_data)

    # Crop thumbnail
    x, y, w, h = target_face.bbox
    pad = int(min(w, h) * 0.15)
    ih, iw = img.shape[:2]
    crop = img[max(0, y - pad):min(ih, y + h + pad), max(0, x - pad):min(iw, x + w + pad)]

    identity_id = db.enroll(name=name, embedding=embedding, thumbnail_img=crop, role=role)
    print(f"[✓] Successfully enrolled '{name}' ({role}) with ID: {identity_id}")
    print(f"    Total identities in database: {len(db.identities)}")


def cmd_enroll_guided(args):
    """Interactive webcam session guiding user through 5 face poses for multi-sample enrollment."""
    name = args.name.strip()
    role = args.role.strip() if args.role else "Member"
    cam_idx = args.camera

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {cam_idx}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = FaceDetector()
    recognizer = FaceRecognizer(backend=args.backend)
    db = FaceDatabase()

    poses_to_capture = [
        ("center", "Step 1/5: Look Straight Ahead (Center)"),
        ("left", "Step 2/5: Turn Head Slightly Left (~15 deg)"),
        ("right", "Step 3/5: Turn Head Slightly Right (~15 deg)"),
        ("up", "Step 4/5: Tilt Head Slightly Upwards"),
        ("expression", "Step 5/5: Smile / Natural Expression")
    ]

    captured_embeddings = []
    captured_crops = []
    current_step = 0
    pose_match_start_time = None
    hold_duration = 1.0  # seconds to hold pose steady before capture

    print(f"\n[*] Starting Guided Multi-Sample Enrollment for '{name}'...")
    print("    Align your face with the target oval and follow on-screen HUD instructions.")
    print("    Press 'q' to abort, 'c' to manually force-capture current pose.\n")

    while current_step < len(poses_to_capture):
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame horizontally for natural selfie view
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        target_pose, prompt_text = poses_to_capture[current_step]

        faces = detector.detect(frame, align=True)
        target_face = faces[0] if faces else None

        guide_res = FacePoseGuide.evaluate(target_face, frame_width=w, frame_height=h, target_pose=target_pose, mirrored=True)

        # Draw Target Oval
        ov = guide_res.target_oval
        oval_color = (0, 255, 0) if guide_res.pose_matched else (255, 200, 0) if guide_res.face_detected else (255, 255, 255)
        cv2.ellipse(frame, (ov["cx"], ov["cy"]), (ov["rx"], ov["ry"]), 0, 0, 360, oval_color, 2, cv2.LINE_AA)

        # Check pose matching timer
        now = time.time()
        capture_triggered = False

        if guide_res.pose_matched:
            if pose_match_start_time is None:
                pose_match_start_time = now
            elapsed = now - pose_match_start_time
            remaining = max(0.0, hold_duration - elapsed)
            countdown_text = f"HOLD STEADY: {remaining:.1f}s"
            cv2.putText(frame, countdown_text, (w // 2 - 120, h - 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            if elapsed >= hold_duration:
                capture_triggered = True
        else:
            pose_match_start_time = None

        # Key press handler
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[!] Enrollment aborted by user.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord('c') and target_face:
            capture_triggered = True

        if capture_triggered and target_face:
            # Capture embedding and crop
            emb = recognizer.extract_embedding(frame, raw_face_data=target_face.raw_face_data)
            captured_embeddings.append(emb)

            x, y, bw, bh = target_face.bbox
            pad = int(min(bw, bh) * 0.15)
            crop = frame[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]
            captured_crops.append(crop)

            print(f"[✓] Captured {prompt_text} (Sample {current_step + 1}/5)")

            # Visual flash feedback
            flash = np.full_like(frame, 255)
            cv2.imshow("Guided Facial Feature Enrollment", flash)
            cv2.waitKey(80)

            current_step += 1
            pose_match_start_time = None
            continue

        # Render Top Banner
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 25, 35), -1)
        cv2.putText(frame, prompt_text, (15, 36), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Render Bottom Guidance
        cv2.rectangle(frame, (0, h - 50), (w, h), (20, 25, 35), -1)
        inst_color = (100, 255, 100) if guide_res.pose_matched else (200, 220, 255)
        cv2.putText(frame, guide_res.instruction, (15, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.62, inst_color, 1, cv2.LINE_AA)

        cv2.imshow("Guided Facial Feature Enrollment", frame)

    cap.release()
    cv2.destroyAllWindows()

    if len(captured_embeddings) >= 3:
        identity_id = db.enroll_multisample(
            name=name,
            embeddings=captured_embeddings,
            thumbnail_imgs=captured_crops,
            role=role
        )
        print("\n" + "=" * 55)
        print("     MULTI-SAMPLE ENROLLMENT COMPLETED SUCCESSFULLY     ")
        print("=" * 55)
        print(f" Identity Name:    {name}")
        print(f" Assigned ID:      {identity_id}")
        print(f" Role / Dept:      {role}")
        print(f" Samples Captured: {len(captured_embeddings)} / 5 facial feature angles")
        print(f" Database Total:   {len(db.identities)} registered identities")
        print("=" * 55 + "\n")
    else:
        print("[!] Enrollment cancelled: Not enough valid samples captured.")


def cmd_identify(args):
    """Identifies all faces in an image against the enrolled database."""
    img_path = args.image
    if not os.path.exists(img_path):
        print(f"[!] Error: Image file not found at '{img_path}'.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Error: Could not decode image at '{img_path}'.")
        return

    detector = FaceDetector()
    recognizer = FaceRecognizer(backend=args.backend)
    db = FaceDatabase()

    print(f"[*] Processing '{img_path}' against database ({len(db.identities)} enrolled)...")
    faces = detector.detect(img, align=True)
    if not faces:
        print("[!] No faces detected.")
        return

    print(f"[+] Found {len(faces)} face(s):")
    annotated = img.copy()

    for idx, face in enumerate(faces, 1):
        embedding = recognizer.extract_embedding(img, raw_face_data=face.raw_face_data)
        matches = db.identify(embedding, threshold=args.threshold, top_k=3)
        best = matches[0]

        status = "IDENTIFIED" if best.is_identified else "UNKNOWN"
        print(f"\n  Face #{idx}: [{status}]")
        print(f"    - Match Name: {best.name} (ID: {best.identity_id})")
        if best.role:
            print(f"    - Role: {best.role}")
        print(f"    - Confidence: {best.confidence:.1f}%")
        print(f"    - Cosine Similarity: {best.cosine_similarity:.4f} (Threshold: {args.threshold})")
        print(f"    - Euclidean Distance: {best.euclidean_distance:.4f}")

        # Draw bbox
        x, y, w, h = face.bbox
        color = (0, 255, 0) if best.is_identified else (0, 0, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"{best.name} ({best.confidence:.0f}%)"
        cv2.putText(annotated, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"\n[+] Annotated result saved to: {args.output}")


def cmd_verify(args):
    """Compares two face images (1:1 verification)."""
    p1, p2 = args.image1, args.image2
    if not os.path.exists(p1) or not os.path.exists(p2):
        print("[!] Error: One or both image files not found.")
        return

    img1 = cv2.imread(p1)
    img2 = cv2.imread(p2)
    if img1 is None or img2 is None:
        print("[!] Error: Could not decode one of the images.")
        return

    detector = FaceDetector()
    recognizer = FaceRecognizer(backend=args.backend, cosine_threshold=args.threshold)

    faces1 = detector.detect(img1, align=True)
    faces2 = detector.detect(img2, align=True)

    if not faces1:
        print(f"[!] No face detected in Image 1: '{p1}'")
        return
    if not faces2:
        print(f"[!] No face detected in Image 2: '{p2}'")
        return

    emb1 = recognizer.extract_embedding(img1, raw_face_data=faces1[0].raw_face_data)
    emb2 = recognizer.extract_embedding(img2, raw_face_data=faces2[0].raw_face_data)

    is_match, cos_score, l2_dist = recognizer.is_match(emb1, emb2, custom_threshold=args.threshold)

    print("\n" + "=" * 50)
    print("           1:1 FACE VERIFICATION REPORT           ")
    print("=" * 50)
    print(f" Image 1: {p1}")
    print(f" Image 2: {p2}")
    print("-" * 50)
    print(f" Verdict:            {'✓ MATCH (SAME PERSON)' if is_match else '✗ NO MATCH (DIFFERENT PERSONS)'}")
    print(f" Cosine Similarity:  {cos_score:.4f} (Threshold: {args.threshold})")
    print(f" Euclidean Distance: {l2_dist:.4f} (Threshold: {recognizer.l2_threshold})")
    print(f" Confidence Score:   {max(0.0, min(100.0, (cos_score / 1.0) * 100)):.1f}%")
    print("=" * 50 + "\n")


def cmd_live(args):
    """Launches live webcam facial recognition stream with interactive OpenCV HUD."""
    cam_idx = args.camera
    print(f"[*] Opening webcam index {cam_idx}...")
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {cam_idx}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    processor = StreamProcessor()
    print("[+] Camera initialized. Press 'q' to quit, 's' to snapshot & enroll.")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break

        annotated, results = processor.process_frame(frame, draw_annotations=True, threshold=args.threshold)

        # FPS calculation
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
        prev_time = curr_time

        cv2.putText(
            annotated,
            f"FPS: {fps:4.1f} | Identities: {len(processor.database.identities)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Facial Recognition & Identification Studio", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def cmd_benchmark(args):
    """Runs speed and latency benchmark on detection and embedding inference."""
    print("=" * 55)
    print("       FACIAL RECOGNITION PIPELINE BENCHMARK       ")
    print("=" * 55)

    detector = FaceDetector()
    recognizer_sface = FaceRecognizer(backend="sface")
    recognizer_pytorch = FaceRecognizer(backend="pytorch")
    db = FaceDatabase()

    runs = args.runs
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple face-like pattern
    cv2.ellipse(test_img, (320, 240), (100, 130), 0, 0, 360, (210, 190, 170), -1)

    print(f"[*] Benchmarking over {runs} iterations on CPU...\n")

    # 1. Detection
    t0 = time.time()
    for _ in range(runs):
        detector.detect(test_img)
    det_time = (time.time() - t0) / runs * 1000.0

    # 2. SFace embedding
    mock_face = np.array([220, 110, 200, 260, 280, 200, 360, 200, 320, 250, 290, 310, 350, 310, 0.99], dtype=np.float32)
    t0 = time.time()
    for _ in range(runs):
        recognizer_sface.extract_embedding(test_img, raw_face_data=mock_face)
    sface_time = (time.time() - t0) / runs * 1000.0

    # 3. PyTorch MobileFaceNet embedding
    crop_112 = np.zeros((112, 112, 3), dtype=np.uint8)
    t0 = time.time()
    for _ in range(runs):
        recognizer_pytorch.extract_embedding(crop_112)
    pt_time = (time.time() - t0) / runs * 1000.0

    # 4. Database 1:N search
    test_emb = np.random.randn(128).astype(np.float32)
    t0 = time.time()
    for _ in range(runs * 5):
        db.identify(test_emb)
    db_time = (time.time() - t0) / (runs * 5) * 1000.0

    print(f"  Component                   | Latency (ms) | Throughput (FPS)")
    print("-" * 55)
    print(f"  YuNet Face Detection        | {det_time:8.2f} ms | {1000.0 / det_time:6.1f} FPS")
    print(f"  SFace Embeddings (ONNX)     | {sface_time:8.2f} ms | {1000.0 / sface_time:6.1f} FPS")
    print(f"  MobileFaceNet (PyTorch CPU) | {pt_time:8.2f} ms | {1000.0 / pt_time:6.1f} FPS")
    print(f"  Database 1:N Search         | {db_time:8.3f} ms | {1000.0 / max(0.001, db_time):6.1f} QPS")
    print("-" * 55)
    total_latency = det_time + sface_time + db_time
    print(f"  End-to-End Pipeline (SFace) | {total_latency:8.2f} ms | {1000.0 / total_latency:6.1f} FPS")
    print("=" * 55 + "\n")


def cmd_list(args):
    """Lists all enrolled identities."""
    db = FaceDatabase()
    identities = db.get_all()
    print(f"\n[*] Enrolled Identities ({len(identities)} total):")
    print("-" * 65)
    print(f" {'ID':<22} | {'Name':<20} | {'Role':<12} | Samples")
    print("-" * 65)
    for item in identities:
        print(f" {item['identity_id']:<22} | {item['name']:<20} | {item['role']:<12} | {item['sample_count']}")
    print("-" * 65 + "\n")


def cmd_delete(args):
    """Deletes an enrolled identity by ID."""
    db = FaceDatabase()
    if db.delete_identity(args.id):
        print(f"[✓] Deleted identity '{args.id}'.")
    else:
        print(f"[!] Identity ID '{args.id}' not found.")


def main():
    parser = argparse.ArgumentParser(description="Face Recognition & Identification CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Enroll
    p_enroll = subparsers.add_parser("enroll", help="Enroll a new identity")
    p_enroll.add_argument("--name", required=True, help="Full name of person")
    p_enroll.add_argument("--image", required=True, help="Path to portrait image")
    p_enroll.add_argument("--role", default="Member", help="Role / department")
    p_enroll.add_argument("--backend", default="sface", choices=["sface", "pytorch"])

    # Enroll Guided (Multi-Sample)
    p_guided = subparsers.add_parser("enroll-guided", help="Interactive 5-pose guided webcam face enrollment")
    p_guided.add_argument("--name", required=True, help="Full name of person")
    p_guided.add_argument("--role", default="Member", help="Role / department")
    p_guided.add_argument("--camera", type=int, default=0, help="Camera index")
    p_guided.add_argument("--backend", default="sface", choices=["sface", "pytorch"])

    # Identify
    p_identify = subparsers.add_parser("identify", help="1:N face identification in an image")
    p_identify.add_argument("--image", required=True, help="Path to input image")
    p_identify.add_argument("--threshold", type=float, default=0.363, help="Cosine threshold (default 0.363)")
    p_identify.add_argument("--output", help="Save annotated image to path")
    p_identify.add_argument("--backend", default="sface", choices=["sface", "pytorch"])

    # Verify
    p_verify = subparsers.add_parser("verify", help="1:1 face verification between two images")
    p_verify.add_argument("--image1", required=True, help="Path to first image")
    p_verify.add_argument("--image2", required=True, help="Path to second image")
    p_verify.add_argument("--threshold", type=float, default=0.363, help="Cosine threshold (default 0.363)")
    p_verify.add_argument("--backend", default="sface", choices=["sface", "pytorch"])

    # Live
    p_live = subparsers.add_parser("live", help="Launch live camera recognition")
    p_live.add_argument("--camera", type=int, default=0, help="Camera device index")
    p_live.add_argument("--threshold", type=float, default=0.363, help="Cosine threshold")

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark detection & embedding speed")
    p_bench.add_argument("--runs", type=int, default=30, help="Number of benchmark iterations")

    # List
    subparsers.add_parser("list", help="List all enrolled identities")

    # Delete
    p_del = subparsers.add_parser("delete", help="Delete an identity by ID")
    p_del.add_argument("--id", required=True, help="Identity ID")

    # Train
    p_train = subparsers.add_parser("train", help="Train PyTorch MobileFaceNet")
    p_train.add_argument("--data-dir", default="", help="Path to dataset directory")
    p_train.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    p_train.add_argument("--synthetic", action="store_true", help="Use synthetic data")

    args = parser.parse_args()

    if args.command == "enroll":
        cmd_enroll(args)
    elif args.command == "enroll-guided":
        cmd_enroll_guided(args)
    elif args.command == "identify":
        cmd_identify(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "live":
        cmd_live(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "train":
        train_model(data_dir=args.data_dir, epochs=args.epochs, use_synthetic=args.synthetic or not args.data_dir)


if __name__ == "__main__":
    main()
