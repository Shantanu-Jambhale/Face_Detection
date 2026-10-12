"""
Face Feature Extraction and Representation Module.
Supports both high-speed OpenCV SFace ONNX backend and PyTorch MobileFaceNet backend.
Computes 128-D L2-normalized face embeddings and metric distances.
"""

import os
import sys
from typing import Optional, Union, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.pytorch_facenet import MobileFaceNet


class FaceRecognizer:
    """
    Extracts discriminative facial feature vectors and calculates pairwise similarities.
    Default backend: SFace (SphereFace2 / ArcFace ONNX model in OpenCV DNN).
    Alternative backend: PyTorch MobileFaceNet.
    """
    def __init__(
        self,
        backend: str = "sface",
        model_path: Optional[str] = None,
        cosine_threshold: float = 0.363,
        l2_threshold: float = 1.128,
        device: str = "cpu"
    ):
        self.backend = backend.lower()
        self.cosine_threshold = cosine_threshold
        self.l2_threshold = l2_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if self.backend == "sface":
            if model_path is None:
                model_path = os.path.join(base_dir, "models", "face_recognition_sface_2021dec.onnx")

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"SFace model not found at '{model_path}'. "
                    f"Run 'python src/download_models.py' to download weights."
                )

            self.model_path = model_path
            self.sface = cv2.FaceRecognizerSF.create(
                model=self.model_path,
                config="",
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
            self.pytorch_model = None

        elif self.backend == "pytorch":
            self.pytorch_model = MobileFaceNet(embedding_size=128).to(self.device)
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.pytorch_model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict):
                    self.pytorch_model.load_state_dict(checkpoint)
            self.pytorch_model.eval()
            self.sface = None
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'sface', 'pytorch'.")

    def extract_embedding(
        self,
        image_or_crop: np.ndarray,
        raw_face_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extracts a normalized 128-dimensional embedding from a face crop or raw image.

        Args:
            image_or_crop: 112x112 aligned face crop, or original BGR image if raw_face_data is provided.
            raw_face_data: 15-float array from YuNet detector (optional, used by SFace alignCrop).

        Returns:
            Normalized 1D numpy array of shape (128,) with float32 values.
        """
        if self.backend == "sface":
            if raw_face_data is not None:
                aligned = self.sface.alignCrop(image_or_crop, raw_face_data)
            else:
                # If already cropped, ensure 112x112 BGR
                aligned = cv2.resize(image_or_crop, (112, 112))
            
            feat = self.sface.feature(aligned)
            feat = feat.flatten().astype(np.float32)
            # Ensure unit L2 norm
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat = feat / norm
            return feat

        else: # pytorch
            if image_or_crop.shape[:2] != (112, 112):
                aligned = cv2.resize(image_or_crop, (112, 112))
            else:
                aligned = image_or_crop

            # Convert BGR (cv2) to RGB, normalize to [-1, 1]
            rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
            tensor = (tensor - 127.5) / 128.0
            tensor = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat_tensor = self.pytorch_model(tensor, normalize=True)
                feat = feat_tensor.cpu().numpy().flatten().astype(np.float32)

            return feat

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[float, float]:
        """
        Computes both Cosine Similarity and Euclidean (L2) distance between two embeddings.

        Returns:
            Tuple of (cosine_similarity, l2_distance)
            cosine_similarity: range [-1.0, 1.0], 1.0 is identical
            l2_distance: range [0.0, 2.0], 0.0 is identical
        """
        e1 = embedding1.flatten()
        e2 = embedding2.flatten()

        # Cosine similarity for unit vectors: dot product
        cosine = float(np.dot(e1, e2))
        cosine = max(-1.0, min(1.0, cosine))

        # L2 Euclidean distance
        l2_dist = float(np.linalg.norm(e1 - e2))

        return cosine, l2_dist

    def is_match(self, embedding1: np.ndarray, embedding2: np.ndarray, custom_threshold: Optional[float] = None) -> Tuple[bool, float, float]:
        """
        Determines if two embeddings belong to the same identity.

        Returns:
            Tuple of (is_match: bool, cosine_score: float, l2_distance: float)
        """
        threshold = custom_threshold if custom_threshold is not None else self.cosine_threshold
        cos_score, l2_score = self.compute_similarity(embedding1, embedding2)
        match = cos_score >= threshold
        return match, cos_score, l2_score
