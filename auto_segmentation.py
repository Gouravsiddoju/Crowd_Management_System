
"""
Automatic Scene Segmentation for Obstacle Detection
Uses SegFormer to identify Floor vs Obstacles (Pillars, Walls, etc.)
"""

import cv2
import numpy as np
import torch
import logging
from typing import Tuple, List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SceneSegmenter:
    """
    Uses a pre-trained Semantic Segmentation model to detect walkable areas.
    """
    
    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512", device: str = None):
        """
        Initialize the segmenter.
        Args:
            model_name: HuggingFace model hub ID
            device: 'cuda' or 'cpu'. If None, auto-detects.
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        
        logger.info(f"Initialized SceneSegmenter on {self.device}")

    def load_model(self):
        """Lazy load the model and processor"""
        if self.model is not None:
            return

        try:
            logger.info(f"Loading model: {self.model_name}...")
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print("Error: 'transformers' library not installed or model load failed.")
            print("Run: pip install transformers")
            raise e

    def segment_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        """
        Segment the frame into Walkable Area (Inclusion) and Obstacles (Exclusions).
        Refined for MAXIMUM AREA INVARIANCE + GRAVITY HEURISTIC:
        - Exclusionary Logic for crowd invariance.
        - Gravity Heuristic: Discards 'floating' floors (ceilings/signage) by keeping only bottom-connected areas.
        """
        self.load_model()
        
        # 1. Inference
        inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Resize logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        
        pred_seg = upsampled_logits.argmax(dim=1)[0].detach().cpu().numpy()
        
        # 2. Exclusionary Logic (Walkable = Not Barrier)
        walkable_mask = np.ones_like(pred_seg, dtype=np.uint8) * 255
        
        # Define Barrier Classes (Things that are definitely NOT floor)
        # 0:wall, 1:building, 2:sky, 4:tree, 5:ceiling, 8:windowpane, 10:cabinet, 
        # 17:plant, 21:signboard, 32:fence, 41:light, 82:column
        barrier_classes = [
            0, 1, 2, 4, 5, 8, 10, 
            15, 17, 19, # Desk, Plant, Commode (?)
            21, # Signboard !!! (Critical for overhead signs)
            30, 32, # Armchair, Fence
            41, # Light/Lamp
            50, # Chandelier
            82, # Column
            116, # Bag (No keep bag as floor) 
            # Add others if needed
        ]
        
        for cls in barrier_classes:
             walkable_mask[pred_seg == cls] = 0
             
        # 3. Clean up
        kernel_close = np.ones((21, 21), np.uint8) 
        walkable_mask = cv2.morphologyEx(walkable_mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = np.ones((9, 9), np.uint8)
        walkable_mask = cv2.morphologyEx(walkable_mask, cv2.MORPH_OPEN, kernel_open)
        
        # 4. GRAVITY HEURISTIC
        # Filter out "floating" walkable areas (like ceiling patches)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(walkable_mask, connectivity=8)
        
        # We only keep the Largest Component AND any component that is "Lower" in the image
        # Ideally, there should be one main floor.
        
        # Find the largest component (idx 0 is background usually, but here background is 0, so idx 1+ are fg)
        if num_labels <= 1:
            logger.warning("No walkable floor detected!")
            return [], []

        # Identify the "Main Floor" candidate
        # Heuristic: Largest area that is in the BOTTOM HALF of the image
        h, w = frame.shape[:2]
        
        best_label = -1
        max_score = -1
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cy = centroids[i][1] # y-centroid
            
            # Score = Area * (How low it is)
            # This favors big areas at the bottom
            # Normalized Y (0 top, 1 bottom)
            norm_y = cy / h
            
            # Penalize floating chunks hard
            if norm_y < 0.3: # Top 30% of screen = Ceiling zone
                score = 0
            else:
                score = area * (norm_y ** 2)
            
            if score > max_score:
                max_score = score
                best_label = i
                
        # Create Final Floor Mask
        final_floor_mask = np.zeros_like(walkable_mask)
        if best_label != -1:
            final_floor_mask[labels == best_label] = 255
            
            # Also merge any other big chunks that are very close to the bottom?
            # For now, let's stick to the single distinct floor plane.
        
        # 5. Extract Inclusions
        contours, hierarchy = cv2.findContours(final_floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inclusion_polygons = []
        full_exclusion_polygons = []
        
        for contour in contours:
            epsilon = 0.003 * cv2.arcLength(contour, True)
            approx_floor = cv2.approxPolyDP(contour, epsilon, True)
            inclusion_polygons.append(approx_floor.reshape(-1, 2).tolist())
            
            # 6. Find Obstacles (Exclusions)
            # Find holes inside this floor
            chunk_mask = np.zeros_like(pred_seg, dtype=np.uint8)
            cv2.drawContours(chunk_mask, [contour], -1, 255, -1)
            
            holes_mask = cv2.bitwise_and(chunk_mask, cv2.bitwise_not(walkable_mask)) # Look at original walkable map for holes
            # Note: We use 'walkable_mask' here not 'final_floor_mask' because 'walkable_mask' has the holes
            
            hole_contours, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in hole_contours:
                if cv2.contourArea(c) < 2000: continue
                
                # Check circularity or shape?
                # Just take it if it's big enough.
                epsilon = 0.005 * cv2.arcLength(c, True)
                approx_hole = cv2.approxPolyDP(c, epsilon, True)
                full_exclusion_polygons.append(approx_hole.reshape(-1, 2).tolist())
        
        return inclusion_polygons, full_exclusion_polygons

    def visualize(self, frame, inclusions, exclusions):
        """Debug visualization"""
        vis = frame.copy()
        
        for inc in inclusions:
            pts = np.array(inc, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            
        for exc in exclusions:
            pts = np.array(exc, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
            cv2.fillPoly(vis, [pts], (0, 0, 100)) # Semi-transparent red-ish fill
            
        return vis

# Quick Test
if __name__ == "__main__":
    segmenter = SceneSegmenter()
    # Mock run would require an image
    print("Segmenter initialized. Import and use 'segment_frame'.")
