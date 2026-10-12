"""
Identity Database and Vector Gallery Manager.
Stores enrolled faces, extracts embeddings, and performs 1:1 verification and 1:N identification.
"""

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class IdentityRecord:
    identity_id: str
    name: str
    role: str
    created_at: str
    sample_count: int
    thumbnail_file: str


@dataclass
class MatchResult:
    identity_id: Optional[str]
    name: str
    role: Optional[str]
    confidence: float          # Similarity percentage / confidence (0 - 100%)
    cosine_similarity: float   # Raw cosine similarity [-1, 1]
    euclidean_distance: float  # L2 distance
    is_identified: bool        # True if >= threshold, False for Unknown
    rank: int


class FaceDatabase:
    """
    Manages vector embeddings and metadata for face identification.
    Saves metadata to database.json and raw embedding vectors to embeddings.npz.
    """
    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            storage_dir = os.path.join(base_dir, "data")

        self.storage_dir = storage_dir
        self.meta_file = os.path.join(storage_dir, "database.json")
        self.embeddings_file = os.path.join(storage_dir, "embeddings.npz")
        self.thumbnails_dir = os.path.join(storage_dir, "thumbnails")

        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.thumbnails_dir, exist_ok=True)

        # In-memory dictionaries:
        # self.identities: {id: IdentityRecord}
        # self.embeddings: {id: List[np.ndarray]}
        # self.centroids: {id: np.ndarray}
        self.identities: Dict[str, IdentityRecord] = {}
        self.embeddings: Dict[str, List[np.ndarray]] = {}
        self.centroids: Dict[str, np.ndarray] = {}

        self.load()

    def load(self):
        """Loads database from disk."""
        if not os.path.exists(self.meta_file) or not os.path.exists(self.embeddings_file):
            return

        try:
            with open(self.meta_file, "r", encoding="utf-8") as f:
                raw_meta = json.load(f)

            npz_data = np.load(self.embeddings_file, allow_pickle=True)

            self.identities.clear()
            self.embeddings.clear()
            self.centroids.clear()

            for id_key, meta in raw_meta.items():
                self.identities[id_key] = IdentityRecord(**meta)
                if id_key in npz_data:
                    emb_list = list(npz_data[id_key])
                    self.embeddings[id_key] = [e.astype(np.float32) for e in emb_list]
                    # Compute normalized centroid
                    avg = np.mean(self.embeddings[id_key], axis=0)
                    norm = np.linalg.norm(avg)
                    self.centroids[id_key] = (avg / norm if norm > 1e-6 else avg).astype(np.float32)
        except Exception as e:
            print(f"[!] Warning: Failed to load existing face database: {e}")

    def save(self):
        """Persists metadata and embeddings to disk."""
        raw_meta = {k: asdict(v) for k, v in self.identities.items()}
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(raw_meta, f, indent=2)

        save_dict = {k: np.array(v, dtype=np.float32) for k, v in self.embeddings.items()}
        np.savez_compressed(self.embeddings_file, **save_dict)

    def enroll(
        self,
        name: str,
        embedding: np.ndarray,
        thumbnail_img: Optional[np.ndarray] = None,
        role: str = "Member",
        identity_id: Optional[str] = None
    ) -> str:
        """
        Registers a new face embedding for an identity.
        If identity already exists, appends the embedding as an additional sample.
        """
        clean_name = name.strip()
        if not identity_id:
            safe_slug = "".join(c if c.isalnum() else "_" for c in clean_name.lower())
            identity_id = f"{safe_slug}_{int(time.time())}"

        # Save thumbnail if provided
        thumb_filename = f"{identity_id}.jpg"
        thumb_path = os.path.join(self.thumbnails_dir, thumb_filename)
        if thumbnail_img is not None and thumbnail_img.size > 0:
            thumb_resized = cv2.resize(thumbnail_img, (160, 160))
            cv2.imwrite(thumb_path, thumb_resized)

        emb_unit = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(emb_unit)
        if norm > 1e-6:
            emb_unit = emb_unit / norm

        if identity_id in self.identities:
            # Append sample
            self.embeddings[identity_id].append(emb_unit)
            self.identities[identity_id].sample_count = len(self.embeddings[identity_id])
            if role:
                self.identities[identity_id].role = role
        else:
            # Create new identity
            rec = IdentityRecord(
                identity_id=identity_id,
                name=clean_name,
                role=role,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                sample_count=1,
                thumbnail_file=thumb_filename if os.path.exists(thumb_path) else ""
            )
            self.identities[identity_id] = rec
            self.embeddings[identity_id] = [emb_unit]

        # Recompute centroid
        avg = np.mean(self.embeddings[identity_id], axis=0)
        c_norm = np.linalg.norm(avg)
        self.centroids[identity_id] = (avg / c_norm if c_norm > 1e-6 else avg).astype(np.float32)

        self.save()
        return identity_id

    def enroll_multisample(
        self,
        name: str,
        embeddings: List[np.ndarray],
        thumbnail_imgs: Optional[List[np.ndarray]] = None,
        role: str = "Member",
        identity_id: Optional[str] = None
    ) -> str:
        """
        Enrolls an identity using multiple face samples captured across angles/expressions.
        Stores multi-exemplar vectors and computes an ensemble centroid.
        """
        clean_name = name.strip()
        if not identity_id:
            safe_slug = "".join(c if c.isalnum() else "_" for c in clean_name.lower())
            identity_id = f"{safe_slug}_{int(time.time())}"

        normed_embs = []
        for e in embeddings:
            e_unit = e.flatten().astype(np.float32)
            n = np.linalg.norm(e_unit)
            if n > 1e-6:
                e_unit = e_unit / n
            normed_embs.append(e_unit)

        # Primary thumbnail (from first sample if available)
        thumb_filename = f"{identity_id}.jpg"
        thumb_path = os.path.join(self.thumbnails_dir, thumb_filename)
        if thumbnail_imgs and len(thumbnail_imgs) > 0 and thumbnail_imgs[0] is not None:
            thumb_resized = cv2.resize(thumbnail_imgs[0], (160, 160))
            cv2.imwrite(thumb_path, thumb_resized)

            # Save auxiliary thumbnails
            for idx, img in enumerate(thumbnail_imgs):
                if img is not None and img.size > 0:
                    sub_thumb = os.path.join(self.thumbnails_dir, f"{identity_id}_sample_{idx}.jpg")
                    cv2.imwrite(sub_thumb, cv2.resize(img, (160, 160)))

        if identity_id in self.identities:
            self.embeddings[identity_id].extend(normed_embs)
            self.identities[identity_id].sample_count = len(self.embeddings[identity_id])
            if role:
                self.identities[identity_id].role = role
        else:
            rec = IdentityRecord(
                identity_id=identity_id,
                name=clean_name,
                role=role,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                sample_count=len(normed_embs),
                thumbnail_file=thumb_filename if os.path.exists(thumb_path) else ""
            )
            self.identities[identity_id] = rec
            self.embeddings[identity_id] = normed_embs

        # Ensemble centroid
        avg = np.mean(self.embeddings[identity_id], axis=0)
        c_norm = np.linalg.norm(avg)
        self.centroids[identity_id] = (avg / c_norm if c_norm > 1e-6 else avg).astype(np.float32)

        self.save()
        return identity_id

    def delete_identity(self, identity_id: str) -> bool:
        """Removes an identity from the database."""
        if identity_id not in self.identities:
            return False

        thumb_file = os.path.join(self.thumbnails_dir, f"{identity_id}.jpg")
        if os.path.exists(thumb_file):
            try:
                os.remove(thumb_file)
            except OSError:
                pass

        del self.identities[identity_id]
        if identity_id in self.embeddings:
            del self.embeddings[identity_id]
        if identity_id in self.centroids:
            del self.centroids[identity_id]

        self.save()
        return True

    def identify(
        self,
        probe_embedding: np.ndarray,
        threshold: float = 0.363,
        top_k: int = 5
    ) -> List[MatchResult]:
        """
        Performs 1:N Identification of a probe face against registered identities.
        Compares against both identity centroids and all individual sample vectors.
        
        Returns:
            Ranked list of MatchResult objects.
        """
        if not self.identities or len(self.centroids) == 0:
            return [MatchResult(
                identity_id=None,
                name="Unknown",
                role=None,
                confidence=0.0,
                cosine_similarity=-1.0,
                euclidean_distance=2.0,
                is_identified=False,
                rank=1
            )]

        p_emb = probe_embedding.flatten().astype(np.float32)
        p_norm = np.linalg.norm(p_emb)
        if p_norm > 1e-6:
            p_emb = p_emb / p_norm

        scored_candidates = []

        for id_key, centroid in self.centroids.items():
            # Check maximum similarity across all sample embeddings of this identity
            sample_embs = self.embeddings.get(id_key, [centroid])
            cos_scores = [float(np.dot(p_emb, s_emb)) for s_emb in sample_embs]
            best_cos = max(cos_scores)
            best_cos = max(-1.0, min(1.0, best_cos))

            # Euclidean distance
            l2_dist = float(np.sqrt(max(0.0, 2.0 * (1.0 - best_cos))))

            # Normalize confidence percentage:
            # threshold -> ~60%, 1.0 -> 100%, 0.0 -> 0%
            if best_cos >= threshold:
                conf = 60.0 + (best_cos - threshold) / max(0.01, (1.0 - threshold)) * 40.0
            else:
                conf = max(0.0, (best_cos / max(0.01, threshold)) * 60.0)

            scored_candidates.append({
                "id": id_key,
                "name": self.identities[id_key].name,
                "role": self.identities[id_key].role,
                "cos": best_cos,
                "l2": l2_dist,
                "confidence": round(conf, 1)
            })

        # Sort descending by cosine similarity
        scored_candidates.sort(key=lambda x: x["cos"], reverse=True)

        results: List[MatchResult] = []
        for rank, cand in enumerate(scored_candidates[:top_k], start=1):
            is_match = cand["cos"] >= threshold
            results.append(MatchResult(
                identity_id=cand["id"] if is_match else None,
                name=cand["name"] if is_match else "Unknown",
                role=cand["role"] if is_match else None,
                confidence=cand["confidence"],
                cosine_similarity=cand["cos"],
                euclidean_distance=cand["l2"],
                is_identified=is_match,
                rank=rank
            ))

        return results

    def get_all(self) -> List[Dict]:
        """Returns list of identity metadata dictionaries with thumbnail paths."""
        out = []
        for id_key, rec in self.identities.items():
            d = asdict(rec)
            d["has_thumbnail"] = bool(rec.thumbnail_file and os.path.exists(os.path.join(self.thumbnails_dir, rec.thumbnail_file)))
            out.append(d)
        return out
