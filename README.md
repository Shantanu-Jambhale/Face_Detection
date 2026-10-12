# NeuralFace Biometric System
### High-Performance Facial Recognition & 1:N Face Identification Engine

A production-grade, state-of-the-art Deep Facial Recognition, 1:1 Verification, and 1:N Identification system built with **PyTorch**, **OpenCV YuNet**, and **SFace (SphereFace2 / ArcFace)**.

---

## 🌟 Key Features

1. **Robust 5-Point Face Detection & Geometric Alignment**:
   - Uses OpenCV's **YuNet** (`face_detection_yunet_2023mar.onnx`) with 5 facial landmarks (eyes, nose, mouth corners).
   - Canonical 2D affine similarity alignment standardizing all faces to canonical $112 \times 112$ pose.
2. **Dual Deep Learning Embedding Backends**:
   - **Production Engine (SFace ONNX)**: Ultra-fast 128-D normalized hypersphere embeddings ($L_2 = 1.0$), running at **50+ FPS on CPU** with <20ms latency.
   - **PyTorch Custom Model (MobileFaceNet + ArcFace)**: Depthwise separable residual bottlenecks, Global Depthwise Convolution (GDConv), and Additive Angular Margin ($m=0.5, s=64.0$) Loss for custom training and fine-tuning.
3. **Identity Vector Gallery Database (`FaceDatabase`)**:
   - Multi-exemplar & centroid representation per identity.
   - 1:1 Verification (`verify(face1, face2)`) with Cosine & Euclidean distances.
   - 1:N Identification (`identify(probe_face, top_k=5)`) with automatic "Unknown" classification below threshold.
   - High-throughput indexing (over 1,000,000 queries/second).
4. **Interactive NeuralFace Studio Web Application**:
   - Modern dark mode interface (`#090D16` obsidian theme, glassmorphism, neon cyan & emerald HUD).
   - **Live Webcam Stream**: Real-time detection bounding boxes, landmarks, confidence metrics, and instant enrollment.
   - **1:1 Face Verification Comparator**: Dual image comparator with circular gauge and match verdict.
   - **1:N Face Identification Search**: Drag & drop multi-face detection and candidate ranking.
   - **Identity Gallery Manager**: View registered identities, face crops, and vector signatures.
   - **Hardware Benchmarks**: Live CPU latency and throughput test suite.
   - **Biometric Attendance Log**: Chronological event tracking exported to `attendance.csv`.
5. **Command Line Interface (`cli.py`)**:
   - Full CLI for batch enrollment, recognition, verification, live camera, and training.

---

## 🚀 Quickstart

### 1. Launch the Interactive Web Studio
```bash
python app.py
```
Open your web browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

---

### 2. Command Line Interface (CLI)

#### List Registered Identities:
```bash
python cli.py list
```

#### 1:N Face Identification in an Image:
```bash
python cli.py identify --image data/sample_portraits/marcus_chen.jpg --output data/identified_result.jpg
```

#### 1:1 Face Verification Between Two Photos:
```bash
python cli.py verify --image1 data/sample_portraits/dr_elena_vance.jpg --image2 data/sample_portraits/marcus_chen.jpg
```

#### Guided 5-Angle Face Feature Enrollment (Webcam):
Walks you through 5 facial poses (Center, Turn Left, Turn Right, Tilt Up, Smile) with on-screen HUD directions and auto-capture:
```bash
python cli.py enroll-guided --name "Your Name" --role "Engineer" --camera 0
```

#### Single Photo Enrollment:
```bash
python cli.py enroll --name "Ada Lovelace" --role "Mathematician" --image path/to/portrait.jpg
```

#### Live Webcam Recognition:
```bash
python cli.py live --camera 0
```

#### Run Hardware Latency Benchmark:
```bash
python cli.py benchmark --runs 30
```

---

## 🧠 Training Custom PyTorch MobileFaceNet

To train or fine-tune MobileFaceNet with ArcFace loss on your own dataset:

```bash
# Organize face images by identity:
# dataset/
#   person_1/img1.jpg, img2.jpg
#   person_2/...

python cli.py train --data-dir path/to/dataset --epochs 10 --batch-size 16
```

To run a pipeline dry-run verification using synthetic face tensors:
```bash
python cli.py train --synthetic --epochs 2
```

Checkpoints are saved to `checkpoints/mobilefacenet_arcface.pth`.

---

## 📊 Latency & Performance Benchmarks (Tested on Host CPU)

| Component | Latency | Throughput |
| :--- | :---: | :---: |
| **YuNet 5-Point Face Detection** | **18.8 ms** | **52.9 FPS** |
| **SFace 128-D Feature Extraction** | **19.5 ms** | **51.1 FPS** |
| **PyTorch MobileFaceNet (CPU)** | **23.3 ms** | **42.8 FPS** |
| **Vector Gallery 1:N Search** | **0.001 ms** | **1,000,000+ QPS** |
| **End-to-End Recognition Pipeline** | **38.4 ms** | **~26 FPS** |

---

## 📂 Project Structure

```
New_Face_Recog/
├── app.py                     # Flask Web Studio Server & REST APIs
├── cli.py                     # Unified CLI for recognition, verification, and live camera
├── models/                    # ONNX model weights
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── checkpoints/               # Trained PyTorch model checkpoints
│   └── mobilefacenet_arcface.pth
├── data/
│   ├── database.json          # Enrolled identity metadata
│   ├── embeddings.npz         # 128-D vector hyperspace store
│   ├── thumbnails/            # Registered face crops
│   ├── sample_portraits/      # Pre-populated demo portraits
│   └── logs/attendance.csv    # Biometric recognition audit log
├── src/
│   ├── detector.py            # YuNet 5-landmark face detector & affine alignment
│   ├── recognizer.py          # SFace & PyTorch MobileFaceNet feature embedder
│   ├── database.py            # FaceDatabase vector gallery engine
│   ├── stream_processor.py    # Real-time multi-face video HUD & temporal smoothing
│   ├── train.py               # PyTorch MobileFaceNet + ArcFace training pipeline
│   ├── download_models.py     # Automated weights downloader
│   └── models/
│       └── pytorch_facenet.py # MobileFaceNet, ArcMarginProduct, TripletLoss
├── static/
│   ├── css/style.css          # Design system & dark mode stylesheet
│   └── js/app.js              # Web studio interactive client logic
├── templates/
│   └── index.html             # Web application dashboard UI
└── tests/
    └── test_pipeline.py       # Unit and integration test suite
```
