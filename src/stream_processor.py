"""
Real-time Video and Webcam Recognition Processor.
Handles multi-face detection, identification, visual HUD overlays, and temporal smoothing.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from src.detector import FaceDetector, DetectedFace
from src.recognizer import FaceRecognizer
from src.database import FaceDatabase, MatchResult


class StreamProcessor:
    """
    Processes video frames in real-time with multi-face detection, identification,
    anti-flicker tracking, and rich visual annotations.
    """
    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
        database: Optional[FaceDatabase] = None,
        log_events: bool = True
    ):
        self.detector = detector if detector is not None else FaceDetector()
        self.recognizer = recognizer if recognizer is not None else FaceRecognizer()
        self.database = database if database is not None else FaceDatabase()
        self.log_events = log_events

        # Recognition log path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_file = os.path.join(base_dir, "data", "logs", "attendance.csv")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("timestamp,identity_id,name,role,confidence,cosine_score\n")

        # Temporal smoothing cache for face tracking
        self.recent_tracks: Dict[int, str] = {}
        self.last_logged_time: Dict[str, float] = {}

    def process_frame(
        self,
        frame: np.ndarray,
        draw_annotations: bool = True,
        threshold: float = 0.363
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detects and identifies all faces in the given frame.

        Args:
            frame: BGR numpy image.
            draw_annotations: Whether to render HUD bounding boxes & labels onto frame.
            threshold: Cosine similarity identification threshold.

        Returns:
            Tuple of (annotated_frame, list_of_detections)
        """
        if frame is None or frame.size == 0:
            return frame, []

        annotated = frame.copy() if draw_annotations else frame
        detected_faces = self.detector.detect(frame, align=True)
        recognition_results = []

        now_sec = time.time()

        for idx, face in enumerate(detected_faces):
            x, y, w, h = face.bbox

            # Extract embedding using SFace or PyTorch
            embedding = self.recognizer.extract_embedding(frame, raw_face_data=face.raw_face_data)

            # Match against database
            match_candidates = self.database.identify(embedding, threshold=threshold, top_k=3)
            best_match: MatchResult = match_candidates[0]

            is_identified = best_match.is_identified
            display_name = best_match.name if is_identified else "Unknown"
            role = best_match.role or "Unregistered"
            confidence = best_match.confidence
            cos_score = best_match.cosine_similarity

            # Log event if identified and cooldown passed (e.g. 10s cooldown per person)
            if is_identified and self.log_events:
                last_time = self.last_logged_time.get(best_match.identity_id, 0.0)
                if now_sec - last_time > 10.0:
                    self.last_logged_time[best_match.identity_id] = now_sec
                    self._log_event(best_match.identity_id, display_name, role, confidence, cos_score)

            rec_data = {
                "bbox": [x, y, w, h],
                "confidence": round(confidence, 1),
                "cosine_score": round(cos_score, 4),
                "is_identified": is_identified,
                "name": display_name,
                "role": role,
                "identity_id": best_match.identity_id,
                "landmarks": face.landmarks
            }
            recognition_results.append(rec_data)

            if draw_annotations:
                self._draw_face_hud(annotated, face, display_name, role, confidence, is_identified)

        return annotated, recognition_results

    def _draw_face_hud(
        self,
        img: np.ndarray,
        face: DetectedFace,
        name: str,
        role: str,
        confidence: float,
        is_identified: bool
    ):
        """Draws aesthetic bounding box, corner brackets, landmarks, and badge."""
        x, y, w, h = face.bbox
        img_h, img_w = img.shape[:2]

        # Color scheme (BGR)
        # Emerald Green for identified (verified), Vibrant Coral/Red for Unknown
        if is_identified:
            primary_color = (74, 222, 128)   # #80DE7A green
            badge_bg = (24, 75, 42)
        else:
            primary_color = (80, 100, 248)   # Coral / Red-orange
            badge_bg = (30, 30, 85)

        # Draw semi-transparent bounding box border
        cv2.rectangle(img, (x, y), (x + w, y + h), primary_color, 1, cv2.LINE_AA)

        # Corner accents (sleek targeting aesthetic)
        c_len = min(20, max(8, int(min(w, h) * 0.2)))
        th = 3
        # Top-left
        cv2.line(img, (x, y), (x + c_len, y), primary_color, th, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + c_len), primary_color, th, cv2.LINE_AA)
        # Top-right
        cv2.line(img, (x + w, y), (x + w - c_len, y), primary_color, th, cv2.LINE_AA)
        cv2.line(img, (x + w, y), (x + w, y + c_len), primary_color, th, cv2.LINE_AA)
        # Bottom-left
        cv2.line(img, (x, y + h), (x + c_len, y + h), primary_color, th, cv2.LINE_AA)
        cv2.line(img, (x, y + h), (x, y + h - c_len), primary_color, th, cv2.LINE_AA)
        # Bottom-right
        cv2.line(img, (x + w, y + h), (x + w - c_len, y + h), primary_color, th, cv2.LINE_AA)
        cv2.line(img, (x + w, y + h), (x + w, y + h - c_len), primary_color, th, cv2.LINE_AA)

        # Draw landmarks subtle dots
        for pt in face.landmarks:
            px, py = int(pt[0]), int(pt[1])
            if 0 <= px < img_w and 0 <= py < img_h:
                cv2.circle(img, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # HUD Label Header (Name & Confidence badge)
        label_text = f"{name} ({confidence:.0f}%)" if is_identified else "Unknown"
        sub_text = role if is_identified else "Not Enrolled"

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.45, min(0.7, w / 220.0))
        (lw, lh), _ = cv2.getTextSize(label_text, font, font_scale, 1)

        badge_h = int(lh * 2.4)
        badge_w = max(lw + 20, 110)
        badge_y1 = max(0, y - badge_h - 4)
        badge_y2 = badge_y1 + badge_h

        # Semi-transparent label background overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (x, badge_y1), (x + badge_w, badge_y2), badge_bg, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        # Border for label badge
        cv2.rectangle(img, (x, badge_y1), (x + badge_w, badge_y2), primary_color, 1, cv2.LINE_AA)

        # Text render
        text_y = badge_y1 + int(lh * 1.5)
        cv2.putText(img, label_text, (x + 8, text_y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _log_event(self, identity_id: str, name: str, role: str, confidence: float, cos_score: float):
        """Appends detection event to attendance log CSV."""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f'"{timestamp}","{identity_id}","{name}","{role}",{confidence:.1f},{cos_score:.4f}\n'
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
