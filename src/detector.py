"""
Face Detection and Geometric Alignment Module.
Uses OpenCV YuNet with 5-landmark localization and canonical affine alignment.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DetectedFace:
    """Represents a detected face with bounding box, landmarks, and confidence score."""
    bbox: Tuple[int, int, int, int]          # x, y, width, height (pixel coordinates)
    confidence: float                        # detection score between 0.0 and 1.0
    landmarks: List[Tuple[float, float]]     # 5 keypoints: [right_eye, left_eye, nose, right_mouth, left_mouth]
    aligned_crop: Optional[np.ndarray]       # 112x112 aligned RGB/BGR face crop
    raw_face_data: np.ndarray                # raw 15-float array from YuNet


# Canonical 5-point face landmarks for standard 112x112 aligned face crop
# (Eye coordinates roughly horizontal and centered)
CANONICAL_5_POINTS_112x112 = np.array([
    [38.2946, 51.6963],   # Right eye
    [73.5318, 51.5014],   # Left eye
    [56.0252, 71.7366],   # Nose tip
    [41.5493, 92.3655],   # Right mouth corner
    [70.7299, 92.2041]    # Left mouth corner
], dtype=np.float32)


def align_face_similarity(image: np.ndarray, landmarks_5pts: np.ndarray, crop_size=(112, 112)) -> np.ndarray:
    """
    Computes optimal similarity transform (rotation + scaling + translation)
    mapping detected 5 landmarks to canonical 112x112 facial positions.
    """
    src_pts = landmarks_5pts.astype(np.float32)
    dst_pts = CANONICAL_5_POINTS_112x112.copy()

    if crop_size != (112, 112):
        scale_x = crop_size[0] / 112.0
        scale_y = crop_size[1] / 112.0
        dst_pts[:, 0] *= scale_x
        dst_pts[:, 1] *= scale_y

    # Compute similarity transform (least-squares 2D affine)
    # estimateAffinePartial2D solves rotation + uniform scale + translation
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        # Fallback to standard affine if partial fails
        M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])

    aligned = cv2.warpAffine(image, M, crop_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return aligned


class FaceDetector:
    """
    High-performance face detector utilizing YuNet ONNX model with 5-point landmarks.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000
    ):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "face_detection_yunet_2023mar.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YuNet model not found at '{model_path}'. "
                f"Run 'python src/download_models.py' to download required weights."
            )

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._current_input_size = (320, 320)

        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self._current_input_size,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def set_score_threshold(self, threshold: float):
        """Update detection confidence threshold."""
        self.conf_threshold = threshold
        self.detector.setScoreThreshold(threshold)

    def detect(self, image: np.ndarray, align: bool = True) -> List[DetectedFace]:
        """
        Detects all faces in an image and extracts bounding boxes, landmarks, and aligned crops.
        
        Args:
            image: BGR numpy image array.
            align: Whether to compute 112x112 canonical aligned crop.
            
        Returns:
            List of DetectedFace dataclass objects.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        
        # Update input size if image dimensions change
        if (w, h) != self._current_input_size:
            self._current_input_size = (w, h)
            self.detector.setInputSize((w, h))

        retval, faces_data = self.detector.detect(image)
        if faces_data is None or len(faces_data) == 0:
            return []

        detected_faces: List[DetectedFace] = []
        for face in faces_data:
            # YuNet output layout:
            # 0:4 -> x, y, w, h
            # 4:6 -> right eye
            # 6:8 -> left eye
            # 8:10 -> nose tip
            # 10:12 -> right mouth corner
            # 12:14 -> left mouth corner
            # 14 -> confidence score
            box_x = int(max(0, face[0]))
            box_y = int(max(0, face[1]))
            box_w = int(min(w - box_x, face[2]))
            box_h = int(min(h - box_y, face[3]))
            score = float(face[14])

            landmarks_raw = np.array([
                [face[4], face[5]],
                [face[6], face[7]],
                [face[8], face[9]],
                [face[10], face[11]],
                [face[12], face[13]]
            ], dtype=np.float32)

            landmarks_list = [(float(pt[0]), float(pt[1])) for pt in landmarks_raw]

            aligned_crop = None
            if align:
                aligned_crop = align_face_similarity(image, landmarks_raw, crop_size=(112, 112))

            detected_faces.append(DetectedFace(
                bbox=(box_x, box_y, box_w, box_h),
                confidence=score,
                landmarks=landmarks_list,
                aligned_crop=aligned_crop,
                raw_face_data=face
            ))

        return detected_faces
