"""
Facial Pose Estimation and Quality Guidance Module.
Analyzes 5 facial keypoints to estimate head yaw (left/right), pitch (up/down),
distance, centering, and image quality for multi-sample biometric enrollment.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from src.detector import DetectedFace


@dataclass
class PoseGuidanceResult:
    face_detected: bool
    pose_matched: bool
    target_pose: str
    estimated_pose: str
    instruction: str
    yaw_ratio: float
    pitch_ratio: float
    distance_status: str       # "good", "too_far", "too_close"
    is_centered: bool
    quality_score: float       # 0.0 - 100.0
    quality_ok: bool
    target_oval: Dict[str, int] # cx, cy, rx, ry


class FacePoseGuide:
    """
    Evaluates detected facial landmarks against target registration poses:
    - 'center': looking straight ahead
    - 'left': turned slightly to the subject's left (nose moves towards right eye in mirrored camera)
    - 'right': turned slightly to the subject's right
    - 'up': tilted slightly upwards
    - 'expression': relaxed or smiling frontal pose
    """

    # Pose calibration thresholds
    YAW_CENTER_MAX = 0.15
    YAW_SIDE_MIN = 0.16
    PITCH_UP_THRESHOLD = 0.48
    PITCH_DOWN_THRESHOLD = 0.65
    PITCH_CENTER_MIN = 0.49
    PITCH_CENTER_MAX = 0.64

    @classmethod
    def evaluate(
        cls,
        face: Optional[DetectedFace],
        frame_width: int,
        frame_height: int,
        target_pose: str = "center",
        mirrored: bool = True
    ) -> PoseGuidanceResult:
        """
        Evaluates face position, pose angle, and quality against requested target pose.
        
        Args:
            face: DetectedFace instance from FaceDetector.
            frame_width: Width of video frame.
            frame_height: Height of video frame.
            target_pose: One of ['center', 'left', 'right', 'up', 'expression'].
            mirrored: True if webcam is mirrored (standard for user interfaces).
        """
        # Recommended target alignment oval in frame center
        target_cx = frame_width // 2
        target_cy = int(frame_height * 0.48)
        target_rx = int(frame_width * 0.22)
        target_ry = int(frame_height * 0.32)
        target_oval = {"cx": target_cx, "cy": target_cy, "rx": target_rx, "ry": target_ry}

        if face is None:
            return PoseGuidanceResult(
                face_detected=False,
                pose_matched=False,
                target_pose=target_pose,
                estimated_pose="none",
                instruction="Position your face inside the target oval",
                yaw_ratio=0.0,
                pitch_ratio=0.5,
                distance_status="too_far",
                is_centered=False,
                quality_score=0.0,
                quality_ok=False,
                target_oval=target_oval
            )

        landmarks = face.landmarks
        r_eye = np.array(landmarks[0])
        l_eye = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        r_mouth = np.array(landmarks[3])
        l_mouth = np.array(landmarks[4])

        # 1. Inter-pupillary distance (IPD)
        ipd = float(np.linalg.norm(l_eye - r_eye))
        ipd_ratio = ipd / max(1, frame_width)

        # Distance check
        if ipd_ratio < 0.12:
            distance_status = "too_far"
        elif ipd_ratio > 0.42:
            distance_status = "too_close"
        else:
            distance_status = "good"

        # 2. Centering check
        x, y, w, h = face.bbox
        face_cx = x + w // 2
        face_cy = y + h // 2
        dx = abs(face_cx - target_cx) / frame_width
        dy = abs(face_cy - target_cy) / frame_height
        is_centered = (dx < 0.18) and (dy < 0.20)

        # 3. Yaw ratio estimation:
        # Distance from nose to left eye vs nose to right eye
        d_left = float(l_eye[0] - nose[0])
        d_right = float(nose[0] - r_eye[0])
        yaw_ratio = float((d_left - d_right) / max(1.0, ipd))

        # 4. Pitch ratio estimation:
        eyes_y = float((r_eye[1] + l_eye[1]) / 2.0)
        mouth_y = float((r_mouth[1] + l_mouth[1]) / 2.0)
        face_height_lm = float(mouth_y - eyes_y)
        pitch_ratio = float((nose[1] - eyes_y) / max(1.0, face_height_lm))

        # Classify estimated pose
        # In mirrored webcam, when user turns to their left,
        # from their perspective it's left, but in image it corresponds to positive yaw_ratio
        if mirrored:
            # mirrored screen view
            user_yaw = yaw_ratio
        else:
            user_yaw = -yaw_ratio

        if pitch_ratio < cls.PITCH_UP_THRESHOLD:
            est_pose = "up"
        elif user_yaw > cls.YAW_SIDE_MIN:
            est_pose = "left"
        elif user_yaw < -cls.YAW_SIDE_MIN:
            est_pose = "right"
        elif cls.PITCH_CENTER_MIN <= pitch_ratio <= cls.PITCH_CENTER_MAX and abs(user_yaw) <= cls.YAW_CENTER_MAX:
            est_pose = "center"
        else:
            est_pose = "neutral"

        # Quality score
        conf_part = face.confidence * 40.0
        dist_part = 30.0 if distance_status == "good" else 10.0
        center_part = 30.0 if is_centered else 10.0
        quality_score = min(100.0, max(0.0, conf_part + dist_part + center_part))
        quality_ok = quality_score >= 65.0

        # Check if target pose matched
        pose_matched = False
        instruction = ""

        if distance_status == "too_far":
            instruction = "Move slightly closer to the camera"
        elif distance_status == "too_close":
            instruction = "Move slightly back from camera"
        elif not is_centered:
            instruction = "Align your face inside the center oval"
        else:
            if target_pose in ("center", "expression"):
                if est_pose == "center" or abs(user_yaw) <= cls.YAW_CENTER_MAX:
                    pose_matched = True
                    instruction = "Perfect! Hold steady looking straight" if target_pose == "center" else "Great! Smile or natural expression"
                elif est_pose == "left":
                    instruction = "Turn head back to center"
                elif est_pose == "right":
                    instruction = "Turn head back to center"
                elif est_pose == "up":
                    instruction = "Lower head slightly to center"

            elif target_pose == "left":
                if est_pose == "left" or user_yaw >= cls.YAW_SIDE_MIN:
                    pose_matched = True
                    instruction = "Pose matched! Hold steady at this angle"
                else:
                    instruction = "Turn your head slightly to your Left (~15°)"

            elif target_pose == "right":
                if est_pose == "right" or user_yaw <= -cls.YAW_SIDE_MIN:
                    pose_matched = True
                    instruction = "Pose matched! Hold steady at this angle"
                else:
                    instruction = "Turn your head slightly to your Right (~15°)"

            elif target_pose == "up":
                if est_pose == "up" or pitch_ratio < cls.PITCH_UP_THRESHOLD:
                    pose_matched = True
                    instruction = "Pose matched! Hold steady looking up"
                else:
                    instruction = "Tilt your head slightly Upwards"

        return PoseGuidanceResult(
            face_detected=True,
            pose_matched=pose_matched and quality_ok,
            target_pose=target_pose,
            estimated_pose=est_pose,
            instruction=instruction,
            yaw_ratio=round(yaw_ratio, 3),
            pitch_ratio=round(pitch_ratio, 3),
            distance_status=distance_status,
            is_centered=is_centered,
            quality_score=round(quality_score, 1),
            quality_ok=quality_ok,
            target_oval=target_oval
        )
