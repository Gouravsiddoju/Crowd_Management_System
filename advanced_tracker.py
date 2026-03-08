"""
Advanced Tracking Module with Re-Identification
Combines YOLO tracking with appearance-based re-ID for robust person tracking
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from deepface import DeepFace
    from scipy.spatial.distance import cosine
    REID_AVAILABLE = True
except ImportError:
    REID_AVAILABLE = False
    print("Warning: DeepFace not available, re-ID disabled")


class PersonTrack:
    """Represents a single person track with Re-ID features"""
    
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], 
                 confidence: float, frame_num: int):
        self.track_id = track_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.first_seen = frame_num
        self.last_seen = frame_num
        self.frames_tracked = 1
        self.times_lost = 0
        
        # Re-ID features
        self.face_embedding = None
        self.appearance_features = []  # Multiple appearance features for robustness
        self.has_valid_reid = False
        
        # Trajectory tracking
        self.trajectory = deque(maxlen=60)  # Last 60 positions
        self.trajectory.append(((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2))
        
        # Quality metrics
        self.avg_confidence = confidence
        self.quality_score = 0.5  # 0-1, updated over time
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float, frame_num: int):
        """Update track with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = frame_num
        self.frames_tracked += 1
        
        # Update trajectory
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.trajectory.append(center)
        
        # Update average confidence
        self.avg_confidence = (self.avg_confidence * 0.9 + confidence * 0.1)
        
        # Update quality score
        self._update_quality_score()
    
    def _update_quality_score(self):
        """Calculate track quality based on multiple factors"""
        # Factor 1: Tracking duration (longer = better)
        duration_score = min(self.frames_tracked / 100, 1.0)
        
        # Factor 2: Average confidence (higher = better)
        conf_score = self.avg_confidence
        
        # Factor 3: Loss frequency (less loss = better)
        loss_score = max(0, 1.0 - (self.times_lost / 10))
        
        # Factor 4: Has Re-ID features (yes = better)
        reid_score = 1.0 if self.has_valid_reid else 0.5
        
        # Weighted combination
        self.quality_score = (
            0.3 * duration_score +
            0.3 * conf_score +
            0.2 * loss_score +
            0.2 * reid_score
        )
    
    def mark_lost(self):
        """Mark track as lost (not detected in current frame)"""
        self.times_lost += 1
    
    def get_velocity(self) -> Tuple[float, float]:
        """Estimate velocity from trajectory"""
        if len(self.trajectory) < 2:
            return (0, 0)
        
        recent = list(self.trajectory)[-10:]  # Last 10 points
        if len(recent) < 2:
            return (0, 0)
        
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        dt = len(recent)
        
        return (dx / dt, dy / dt)
    
    def predict_position(self, frames_ahead: int = 5) -> Tuple[int, int]:
        """Predict future position based on trajectory"""
        if len(self.trajectory) < 2:
            return self.trajectory[-1] if self.trajectory else (0, 0)
        
        vx, vy = self.get_velocity()
        last_pos = self.trajectory[-1]
        
        pred_x = int(last_pos[0] + vx * frames_ahead)
        pred_y = int(last_pos[1] + vy * frames_ahead)
        
        return (pred_x, pred_y)

    def duration_seconds(self, fps: float) -> float:
        """Calculate duration of track in seconds"""
        if fps <= 0:
            return 0.0
        return (self.last_seen - self.first_seen) / fps


class ReIDMatcher:
    """Re-identification matcher using face and appearance features"""
    
    def __init__(self, config: dict):
        self.enabled = REID_AVAILABLE and config.get('reid_enabled', True)
        self.face_threshold = config.get('reid_threshold', 0.6)
        self.appearance_threshold = 0.7
        self.reid_interval = config.get('reid_interval', 5)
        
        if not self.enabled and config.get('reid_enabled', True):
            print("Warning: Re-ID requested but dependencies not available")
    
    def extract_face_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract face embedding from person bounding box"""
        if not self.enabled:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            person_height = y2 - y1
            
            # Extract head region (top 40% of person)
            head_y2 = min(y1 + int(person_height * 0.4), y2)
            head_region = frame[max(0, y1):head_y2, max(0, x1):min(x2, frame.shape[1])]
            
            if head_region.size == 0 or head_region.shape[0] < 40 or head_region.shape[1] < 40:
                return None
            
            # Generate embedding
            embedding_objs = DeepFace.represent(
                head_region,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend="opencv"
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                return np.array(embedding_objs[0]["embedding"])
        
        except Exception as e:
            pass
        
        return None
    
    def extract_appearance_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract appearance features (color histogram) from person bounding box"""
        try:
            import cv2
            x1, y1, x2, y2 = bbox
            
            # Extract person region
            person_region = frame[max(0, y1):min(y2, frame.shape[0]), 
                                 max(0, x1):min(x2, frame.shape[1])]
            
            if person_region.size == 0:
                return None
            
            # Convert to HSV for more robust color matching
            hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
            
            # Normalize
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            # Concatenate
            features = np.concatenate([hist_h, hist_s, hist_v])
            
            return features
        
        except Exception as e:
            return None
    
    def match_face(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate face similarity"""
        try:
            similarity = 1 - cosine(emb1, emb2)
            return max(0, min(1, similarity))
        except:
            return 0
    
    def match_appearance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate appearance similarity using histogram correlation"""
        try:
            import cv2
            similarity = cv2.compareHist(
                feat1.astype(np.float32),
                feat2.astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            return max(0, min(1, similarity))
        except:
            return 0
    
    def find_best_match(self, track: PersonTrack, lost_tracks: List[PersonTrack]) -> Optional[Tuple[PersonTrack, float]]:
        """Find best matching lost track for re-identification"""
        if not lost_tracks or not track.has_valid_reid:
            return None
        
        best_match = None
        best_score = 0
        
        for lost_track in lost_tracks:
            if not lost_track.has_valid_reid:
                continue
            
            # Calculate combined similarity
            face_sim = 0
            appearance_sim = 0
            
            # Face similarity
            if track.face_embedding is not None and lost_track.face_embedding is not None:
                face_sim = self.match_face(track.face_embedding, lost_track.face_embedding)
            
            # Appearance similarity (average of multiple features)
            if track.appearance_features and lost_track.appearance_features:
                app_sims = []
                for feat1 in track.appearance_features[-3:]:  # Last 3 features
                    for feat2 in lost_track.appearance_features[-3:]:
                        app_sims.append(self.match_appearance(feat1, feat2))
                if app_sims:
                    appearance_sim = np.mean(app_sims)
            
            # Combined score (Balance updated for surveillance - less reliance on face)
            if face_sim > 0.4:  # Only use face if similarity is decent
                combined_score = 0.4 * face_sim + 0.6 * appearance_sim
            else:
                combined_score = appearance_sim
            
            # Check threshold and update best match
            if combined_score > 0.65 and combined_score > best_score:  # Stricter combined threshold
                best_score = combined_score
                best_match = lost_track
        
        if best_match:
            return (best_match, best_score)
        
        return None


class TrackManager:
    """Manages all person tracks with Re-ID"""
    
    def __init__(self, config: dict):
        self.active_tracks: Dict[int, PersonTrack] = {}
        self.lost_tracks: List[PersonTrack] = []
        self.completed_tracks: List[PersonTrack] = []
        
        self.max_age = config.get('max_age', 90)  # Frames to keep lost tracks
        self.reid_matcher = ReIDMatcher(config)
        
        self.next_id = 1
        self.unique_ids = set()
        
        # Statistics
        self.total_detections = 0
        self.total_reid_matches = 0
    
    def update(self, detections: List[Tuple], frame: np.ndarray, frame_num: int, 
               yolo_tracks: Optional[Dict] = None) -> Dict[int, PersonTrack]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of (bbox, confidence) tuples
            frame: Current frame for Re-ID
            frame_num: Current frame number
            yolo_tracks: Optional dict of {yolo_id: (bbox, conf)} from YOLO tracker
            
        Returns:
            Dictionary of active tracks
        """
        self.total_detections += len(detections)
        
        # If YOLO provides track IDs, use them
        if yolo_tracks:
            return self._update_with_yolo_tracks(yolo_tracks, frame, frame_num)
        
        # Otherwise, manual tracking (less common)
        return self._update_manual_tracking(detections, frame, frame_num)
    
    def _update_with_yolo_tracks(self, yolo_tracks: Dict, frame: np.ndarray, frame_num: int) -> Dict[int, PersonTrack]:
        """Update using YOLO's built-in tracking"""
        updated_ids = set()
        
        for yolo_id, (bbox, conf) in yolo_tracks.items():
            # Check if this YOLO ID already has a track
            if yolo_id in self.active_tracks:
                # Update existing track
                track = self.active_tracks[yolo_id]
                track.update(bbox, conf, frame_num)
                updated_ids.add(yolo_id)
                
                # Extract Re-ID features periodically
                if frame_num % self.reid_matcher.reid_interval == 0:
                    self._extract_reid_features(track, frame)
            
            else:
                # New track (or re-identified track)
                # Check if this could be a lost track
                new_track = PersonTrack(yolo_id, bbox, conf, frame_num)
                self._extract_reid_features(new_track, frame)
                
                # Try to match with lost tracks
                if new_track.has_valid_reid and self.lost_tracks:
                    match_result = self.reid_matcher.find_best_match(new_track, self.lost_tracks)
                    
                    if match_result:
                        # Re-identified!
                        matched_track, score = match_result
                        original_id = matched_track.track_id
                        
                        # Transfer Re-ID features and history
                        new_track.track_id = original_id
                        new_track.first_seen = matched_track.first_seen
                        new_track.frames_tracked = matched_track.frames_tracked + 1
                        new_track.times_lost = matched_track.times_lost
                        new_track.trajectory.extend(matched_track.trajectory)
                        
                        # Remove from lost tracks
                        self.lost_tracks.remove(matched_track)
                        
                        self.total_reid_matches += 1
                        # print(f"✓ Re-ID: Matched track {original_id} (score: {score:.2f})")
                
                self.active_tracks[yolo_id] = new_track
                self.unique_ids.add(yolo_id)
                updated_ids.add(yolo_id)
        
        # Move non-updated tracks to lost
        lost_ids = set(self.active_tracks.keys()) - updated_ids
        for lost_id in lost_ids:
            track = self.active_tracks.pop(lost_id)
            track.mark_lost()
            self.lost_tracks.append(track)
        
        # Clean up old lost tracks
        self._cleanup_lost_tracks(frame_num)
        
        return self.active_tracks
    
    def _update_manual_tracking(self, detections: List, frame: np.ndarray, frame_num: int) -> Dict[int, PersonTrack]:
        """Manual tracking without YOLO track IDs (fallback)"""
        # Simplified: just create new tracks for each detection
        # For production, implement proper IoU-based matching
        
        self.active_tracks = {}
        
        for bbox, conf in detections:
            track_id = self.next_id
            self.next_id += 1
            
            track = PersonTrack(track_id, bbox, conf, frame_num)
            self._extract_reid_features(track, frame)
            
            self.active_tracks[track_id] = track
            self.unique_ids.add(track_id)
        
        return self.active_tracks
    
    def _extract_reid_features(self, track: PersonTrack, frame: np.ndarray):
        """Extract Re-ID features for a track"""
        # Extract face embedding
        face_emb = self.reid_matcher.extract_face_embedding(frame, track.bbox)
        if face_emb is not None:
            track.face_embedding = face_emb
            track.has_valid_reid = True
        
        # Extract appearance features
        app_feat = self.reid_matcher.extract_appearance_features(frame, track.bbox)
        if app_feat is not None:
            track.appearance_features.append(app_feat)
            # Keep only last 5 appearance features
            track.appearance_features = track.appearance_features[-5:]
            track.has_valid_reid = True
    
    def _cleanup_lost_tracks(self, current_frame: int):
        """Remove old lost tracks"""
        # Move very old tracks to completed FIRST
        for track in list(self.lost_tracks):
            if (current_frame - track.last_seen) >= self.max_age:
                self.completed_tracks.append(track)

        # Then filter keeping only active ones
        self.lost_tracks = [
            track for track in self.lost_tracks
            if (current_frame - track.last_seen) < self.max_age
        ]
    
    def get_statistics(self) -> dict:
        """Get tracking statistics"""
        return {
            'active_tracks': len(self.active_tracks),
            'lost_tracks': len(self.lost_tracks),
            'unique_ids': len(self.unique_ids),
            'total_detections': self.total_detections,
            'reid_matches': self.total_reid_matches,
            'reid_success_rate': (
                self.total_reid_matches / max(1, len(self.lost_tracks) + self.total_reid_matches)
            ) * 100
        }
