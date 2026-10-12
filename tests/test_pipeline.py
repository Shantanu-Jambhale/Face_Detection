"""
Unit and Integration Test Suite for NeuralFace Biometric System.
Verifies detector, embedding extractor, PyTorch MobileFaceNet,
gallery database, and end-to-end identification pipeline.
"""

import os
import sys
import unittest
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector import FaceDetector, align_face_similarity
from src.recognizer import FaceRecognizer
from src.database import FaceDatabase
from src.stream_processor import StreamProcessor
from src.models.pytorch_facenet import MobileFaceNet, ArcMarginProduct, TripletLoss
from src.pose_guide import FacePoseGuide


class TestFacePipeline(unittest.TestCase):

    def setUp(self):
        self.detector = FaceDetector()
        self.recognizer_sface = FaceRecognizer(backend="sface")
        self.recognizer_pt = FaceRecognizer(backend="pytorch")
        self.db = FaceDatabase(storage_dir=os.path.join(PROJECT_ROOT, "tests", "test_data"))

    def tearDown(self):
        # Cleanup test database folder
        test_dir = os.path.join(PROJECT_ROOT, "tests", "test_data")
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_01_models_exist(self):
        """Verifies required model weights are present."""
        models_dir = os.path.join(PROJECT_ROOT, "models")
        yunet_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        sface_path = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")

        self.assertTrue(os.path.exists(yunet_path), "YuNet model file missing")
        self.assertTrue(os.path.exists(sface_path), "SFace model file missing")
        self.assertGreater(os.path.getsize(yunet_path), 200_000)
        self.assertGreater(os.path.getsize(sface_path), 35_000_000)

    def test_02_pytorch_mobilefacenet(self):
        """Verifies PyTorch MobileFaceNet backbone and ArcFace loss."""
        model = MobileFaceNet(embedding_size=128)
        dummy_in = torch.randn(2, 3, 112, 112)
        out = model(dummy_in, normalize=True)

        self.assertEqual(out.shape, (2, 128))
        # Norms should be 1.0
        norms = torch.norm(out, dim=1).detach().numpy()
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

        # ArcMarginProduct
        arc = ArcMarginProduct(in_features=128, out_features=5, scale=32.0, margin=0.5)
        labels = torch.tensor([0, 1])
        logits = arc(out, labels)
        self.assertEqual(logits.shape, (2, 5))

    def test_03_sface_embedding_extraction(self):
        """Verifies SFace extracts 128-D normalized embedding vector."""
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        crop[30:80, 30:80] = 180  # bright region

        emb = self.recognizer_sface.extract_embedding(crop)
        self.assertEqual(emb.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=4)

    def test_04_similarity_computation(self):
        """Verifies cosine similarity and Euclidean L2 distance calculations."""
        v1 = np.random.randn(128).astype(np.float32)
        v1 /= np.linalg.norm(v1)

        # Self similarity must be 1.0 and distance 0.0
        cos, l2 = self.recognizer_sface.compute_similarity(v1, v1)
        self.assertAlmostEqual(cos, 1.0, places=4)
        self.assertAlmostEqual(l2, 0.0, places=4)

        # Orthogonal vectors should have ~0.0 cosine and ~sqrt(2) distance
        v2 = np.zeros(128, dtype=np.float32)
        v2[0] = 1.0
        v3 = np.zeros(128, dtype=np.float32)
        v3[1] = 1.0
        cos_orth, l2_orth = self.recognizer_sface.compute_similarity(v2, v3)
        self.assertAlmostEqual(cos_orth, 0.0, places=4)
        self.assertAlmostEqual(l2_orth, np.sqrt(2.0), places=4)

    def test_05_database_enroll_identify_delete(self):
        """Verifies database enrollment, 1:N identification, and deletion."""
        test_emb = np.random.randn(128).astype(np.float32)
        test_emb /= np.linalg.norm(test_emb)

        # Enroll test identity
        test_id = self.db.enroll(name="Test Subject Alpha", embedding=test_emb, role="QA Engineer")
        self.assertIn(test_id, self.db.identities)
        self.assertEqual(self.db.identities[test_id].name, "Test Subject Alpha")

        # Identify with exact embedding
        results = self.db.identify(test_emb, threshold=0.363, top_k=1)
        self.assertTrue(results[0].is_identified)
        self.assertEqual(results[0].name, "Test Subject Alpha")
        self.assertAlmostEqual(results[0].cosine_similarity, 1.0, places=4)

        # Identify with distant random vector -> Unknown
        random_emb = -test_emb  # opposite direction
        unknown_results = self.db.identify(random_emb, threshold=0.363, top_k=1)
        self.assertFalse(unknown_results[0].is_identified)
        self.assertEqual(unknown_results[0].name, "Unknown")

        # Delete identity
        deleted = self.db.delete_identity(test_id)
        self.assertTrue(deleted)
        self.assertNotIn(test_id, self.db.identities)

    def test_06_1to1_verification(self):
        """Verifies 1:1 match logic."""
        v1 = np.random.randn(128).astype(np.float32)
        v1 /= np.linalg.norm(v1)

        # Same vector -> match True
        is_match, cos, l2 = self.recognizer_sface.is_match(v1, v1, custom_threshold=0.363)
        self.assertTrue(is_match)

        # Opposite vector -> match False
        is_match, cos, l2 = self.recognizer_sface.is_match(v1, -v1, custom_threshold=0.363)
        self.assertFalse(is_match)

    def test_07_multisample_enrollment(self):
        """Verifies multi-sample face enrollment and ensemble matching."""
        # Create 3 slightly varied embeddings simulating angles for the same person
        base = np.random.randn(128).astype(np.float32)
        base /= np.linalg.norm(base)

        e1 = base.copy() # Center
        e2 = base + np.random.randn(128).astype(np.float32) * 0.1 # Left angle
        e2 /= np.linalg.norm(e2)
        e3 = base + np.random.randn(128).astype(np.float32) * 0.1 # Right angle
        e3 /= np.linalg.norm(e3)

        mock_crop = np.zeros((112, 112, 3), dtype=np.uint8)
        identity_id = self.db.enroll_multisample(
            name="Multi Sample Subject",
            embeddings=[e1, e2, e3],
            thumbnail_imgs=[mock_crop, mock_crop, mock_crop],
            role="Research Fellow"
        )

        self.assertIn(identity_id, self.db.identities)
        self.assertEqual(self.db.identities[identity_id].sample_count, 3)
        self.assertEqual(len(self.db.embeddings[identity_id]), 3)

        # Identify using probe matching e2
        results = self.db.identify(e2, threshold=0.363, top_k=1)
        self.assertTrue(results[0].is_identified)
        self.assertEqual(results[0].name, "Multi Sample Subject")
        self.assertGreater(results[0].cosine_similarity, 0.95)

    def test_08_pose_guide(self):
        """Verifies pose guidance evaluation."""
        from src.detector import DetectedFace

        # Synthetic frontal face landmarks in 640x480 frame
        # [r_eye, l_eye, nose, r_mouth, l_mouth]
        mock_landmarks = [
            (280.0, 200.0), # Right eye
            (360.0, 200.0), # Left eye
            (320.0, 240.0), # Nose tip
            (290.0, 270.0), # Right mouth
            (350.0, 270.0)  # Left mouth
        ]
        face = DetectedFace(
            bbox=(250, 160, 140, 150),
            confidence=0.98,
            landmarks=mock_landmarks,
            aligned_crop=None,
            raw_face_data=np.zeros(15, dtype=np.float32)
        )

        res = FacePoseGuide.evaluate(face, frame_width=640, frame_height=480, target_pose="center")
        self.assertTrue(res.face_detected)
        self.assertTrue(res.is_centered)
        self.assertEqual(res.estimated_pose, "center")
        self.assertTrue(res.pose_matched)

    def test_09_stream_frame_endpoint_live(self):
        """Verifies /api/stream_frame endpoint returns lean detections without annotated frame when requested."""
        import base64
        import cv2
        from app import app

        client = app.test_client()
        # Create a synthetic 640x480 test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        resp = client.post("/api/stream_frame", json={
            "image_b64": b64,
            "threshold": 0.363,
            "include_annotated": False
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("detections", data)
        self.assertIn("latency_ms", data)
        self.assertIsNone(data["annotated_frame"])

    def test_10_stream_frame_endpoint_annotated(self):
        """Verifies /api/stream_frame endpoint returns annotated frame when requested."""
        import base64
        import cv2
        from app import app

        client = app.test_client()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        resp = client.post("/api/stream_frame", json={
            "image_b64": b64,
            "threshold": 0.363,
            "include_annotated": True
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIsNotNone(data["annotated_frame"])
        self.assertTrue(data["annotated_frame"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
