import numpy as np
import cv2
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple, Set
from collections import defaultdict

# Import PersonTrack from existing module
try:
    from advanced_tracker import PersonTrack
except ImportError:
    # Fallback mock for development/testing if file not found in path
    class PersonTrack:
        pass

class CrossCameraMatcher:
    """
    Matches people across different cameras/videos using Re-ID features.
    """
    
    def __init__(self, match_threshold: float = 0.3):
        self.match_threshold = match_threshold
        # Stores tracks: {video_name: [PersonTrack, ...]}
        self.video_tracks: Dict[str, List[PersonTrack]] = defaultdict(list)
        # Global ID mapping: {(video_name, local_track_id) -> global_id}
        self.global_id_map: Dict[Tuple[str, int], int] = {}
        self.next_global_id = 1
        
    def add_video_tracks(self, video_name: str, tracks: List[PersonTrack]):
        """Add all unique tracks found in a video"""
        # Filter out tracks with poor quality or no features
        valid_tracks = []
        for t in tracks:
            # Must have at least one appearance feature or face embedding
            if t.has_valid_reid or t.face_embedding is not None or len(t.appearance_features) > 0:
                valid_tracks.append(t)
        
        print(f"  Received {len(tracks)} tracks from {video_name}, kept {len(valid_tracks)} valid for Re-ID")
        self.video_tracks[video_name] = valid_tracks

    def match_tracks(self) -> Tuple[int, Dict[Tuple[str, int], int]]:
        """
        Perform clustering to assign global IDs.
        Returns:
            total_unique_people (int)
            global_id_map (dict)
        """
        videos = sorted(list(self.video_tracks.keys()))
        if not videos:
            return 0, {}
        
        # Initialize Union-Find structure
        # Elements are (video_name, local_track_id)
        parent = {}
        
        def find(item):
            if parent[item] != item:
                parent[item] = find(parent[item])
            return parent[item]
        
        def union(item1, item2):
            root1 = find(item1)
            root2 = find(item2)
            if root1 != root2:
                parent[root1] = root2
                return True
            return False
            
        # Initialize all tracks as their own sets
        all_track_items = []
        for vid in videos:
            for track in self.video_tracks[vid]:
                item = (vid, track.track_id)
                parent[item] = item
                all_track_items.append((item, track))
                
        # Compare tracks across different videos
        # We only compare tracks from different videos
        print("\nRunning Global Matching...")
        
        # Greedy matching strategy:
        # Compare every track from Video i with every track from Video j
        # Calculate similarity scores
        # Merge if similarity > threshold
        
        # Optimization: Use representative features (average) for comparison
        # Pre-calculate normalized features for efficiency
        
        comparisons = 0
        matches_found = 0
        
        for i in range(len(videos)):
            for j in range(i + 1, len(videos)):
                vid1 = videos[i]
                vid2 = videos[j]
                
                tracks1 = self.video_tracks[vid1]
                tracks2 = self.video_tracks[vid2]
                
                print(f"  Comparing {vid1} ({len(tracks1)} tracks) <-> {vid2} ({len(tracks2)} tracks)...")
                
                # Perform pairwise comparison
                # Note: For very large numbers of tracks, this is O(N*M), might be slow.
                # Given user mentions 4 videos, likely < 500 people per video, so < 250k comparisons. Tolerable.
                
                for t1 in tracks1:
                    for t2 in tracks2:
                        sim = self._calculate_similarity(t1, t2)
                        
                        if sim > self.match_threshold:
                            item1 = (vid1, t1.track_id)
                            item2 = (vid2, t2.track_id)
                            if union(item1, item2):
                                matches_found += 1
                                # print(f"    Match found: {vid1} ID:{t1.track_id} <-> {vid2} ID:{t2.track_id} (Sim: {sim:.2f})")
        
        print(f"  Total pairwise matches established: {matches_found}")
        
        # Assign Global IDs based on sets
        unique_sets = defaultdict(list)
        for vid in videos:
            for track in self.video_tracks[vid]:
                item = (vid, track.track_id)
                root = find(item)
                unique_sets[root].append(item)
        
        self.next_global_id = 1
        self.global_id_map = {}
        
        for root, items in unique_sets.items():
            gid = self.next_global_id
            self.next_global_id += 1
            for item in items:
                self.global_id_map[item] = gid
                
        total_unique = self.next_global_id - 1
        return total_unique, self.global_id_map

    def _calculate_similarity(self, t1: PersonTrack, t2: PersonTrack) -> float:
        """Calculate similarity between two tracks"""
        
        # 1. Face Similarity (Strongest signal)
        face_sim = 0
        if t1.face_embedding is not None and t2.face_embedding is not None:
             face_sim = max(0, 1 - cosine(t1.face_embedding, t2.face_embedding))
             if face_sim > 0.6: # High confidence face match
                 return face_sim
        
        # 2. Appearance Similarity
        # Compare average histograms
        app_sim = 0
        if t1.appearance_features and t2.appearance_features:
            # Strategy: Compare best pair of features (max-max strategy) or average
            # Let's try max-max for robustness against occlusion/lighting changes
            
            # Using only last 3 features for efficiency
            feats1 = t1.appearance_features[-3:]
            feats2 = t2.appearance_features[-3:]
            
            scores = []
            for f1 in feats1:
                for f2 in feats2:
                    try:
                        score = cv2.compareHist(
                            f1.astype(np.float32), 
                            f2.astype(np.float32), 
                            cv2.HISTCMP_CORREL
                        )
                        scores.append(score)
                    except:
                        pass
            
            if scores:
                app_sim = np.mean(scores) # Take average for consistency (prevents single-frame false positives)
                
        # Combined Score
        if face_sim > 0.6: # Strong face match overrides appearance
             return 0.7 * face_sim + 0.3 * app_sim
        elif face_sim > 0: # Weak face match helps but relies on appearance
             return 0.3 * face_sim + 0.7 * app_sim
        else:
            return app_sim
