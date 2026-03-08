"""
Preprocessing Module for Environmental Robustness
Handles varying lighting, motion blur, noise, and quality issues
For Indian railway station and bus stand footage
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import warnings
import sys
import os

from vision_enhance_v2_5.enhancer import VisionEnhancer

warnings.filterwarnings('ignore')


class FramePreprocessor:
    """
    Preprocessing pipeline for robust person detection in challenging conditions:
    - Varying lighting (day/night, shadows, direct sunlight)
    - Motion blur (moving camera, fast motion)
    - Low quality (compression artifacts, noise)
    - Weather conditions (rain, fog)
    """
    
    def __init__(self, config: dict, device: str = 'cuda'):
        self.enabled = config.get('enabled', True)
        self.use_v2_5_enhancer = config.get('use_v2_5_enhancer', False)
        self.enhancer_temporal = config.get('enhancer_temporal', True)
        self.enhancer_exposure = config.get('enhancer_exposure', True)
        self.enhancer_calibrated = config.get('enhancer_calibrated', True)
        
        self.lighting_norm = config.get('lighting_normalization', True)
        self.motion_blur_handling = config.get('motion_blur_handling', False)
        self.denoising = config.get('denoising', True)
        self.auto_white_balance = config.get('auto_white_balance', True)
        self.stabilization = config.get('stabilization', False)
        
        # Initialize VisionEnhancer v2.5 if enabled
        if self.use_v2_5_enhancer:
            try:
                self.enhancer = VisionEnhancer(
                    device=device,
                    temporal=self.enhancer_temporal,
                    exposure_comp=self.enhancer_exposure,
                    hdr_calibrated=self.enhancer_calibrated,
                    verbose=False
                )
                print("✓ Advanced Vision Enhancer v2.5 initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize VisionEnhancer v2.5 ({e}). Falling back to standard preprocessing.")
                self.use_v2_5_enhancer = False
                self.enhancer = None
        
        # CLAHE for lighting normalization
        if self.lighting_norm:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # For video stabilization
        self.prev_frame = None
        self.transform_history = []
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Preprocessed frame (BGR)
        """
        if not self.enabled:
            return frame
        
        if self.use_v2_5_enhancer and hasattr(self, 'enhancer') and self.enhancer:
            return self.enhancer.enhance(frame)
            
        processed = frame.copy()
        
        # 1. Denoising (remove compression artifacts)
        if self.denoising:
            processed = self._denoise(processed)
        
        # 2. Auto white balance (color consistency)
        if self.auto_white_balance:
            processed = self._white_balance(processed)
        
        # 3. Lighting normalization (handle day/night/shadows)
        if self.lighting_norm:
            processed = self._normalize_lighting(processed)
        
        # 4. Motion blur handling (if enabled)
        if self.motion_blur_handling:
            processed = self._handle_motion_blur(processed)
        
        # 5. Video stabilization (if enabled)
        if self.stabilization:
            processed = self._stabilize(processed)
        
        return processed
    
    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Remove noise and compression artifacts using Non-Local Means Denoising
        Fast approximation for real-time processing
        """
        return cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 15)
    
    def _white_balance(self, frame: np.ndarray) -> np.ndarray:
        """
        Simple white balance using gray world assumption
        Helps maintain color consistency across varying lighting
        """
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    def _normalize_lighting(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize lighting using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Applied to luminance channel to avoid color distortion
        Excellent for low-light and high-contrast scenes
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        
        # Merge and convert back to BGR
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _handle_motion_blur(self, frame: np.ndarray) -> np.ndarray:
        """
        Reduce motion blur using Wiener filter approximation
        Note: Computationally expensive, only use if necessary
        """
        # Simple sharpening kernel as approximation
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        
        # Blend with original to avoid over-sharpening
        result = cv2.addWeighted(frame, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def _stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Simple digital video stabilization using frame-to-frame motion estimation
        Reduces jitter from handheld or vibrating cameras
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(self.prev_frame, maxCorners=200,
                                           qualityLevel=0.01, minDistance=30)
        
        if prev_pts is None:
            self.prev_frame = curr_gray
            return frame
        
        # Calculate optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, curr_gray,
                                                        prev_pts, None)
        
        # Filter valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        if len(prev_pts) < 10:
            self.prev_frame = curr_gray
            return frame
        
        # Estimate transform
        transform = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        
        if transform is None:
            self.prev_frame = curr_gray
            return frame
        
        # Apply smoothed transform
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(frame, transform, (w, h))
        
        self.prev_frame = curr_gray
        
        return stabilized
    
    def reset(self):
        """Reset preprocessor state (for new video/stream)"""
        self.prev_frame = None
        self.transform_history = []


class AdaptiveThresholder:
    """
    Dynamically adjust detection confidence threshold based on lighting conditions
    Helps maintain consistent detection quality across day/night cycles
    """
    
    def __init__(self, base_threshold: float = 0.25, 
                 min_threshold: float = 0.15,
                 max_threshold: float = 0.5):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.brightness_history = []
        self.history_size = 30
    
    def get_adaptive_threshold(self, frame: np.ndarray) -> float:
        """
        Calculate adaptive confidence threshold based on frame brightness
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Adjusted confidence threshold
        """
        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Update history
        self.brightness_history.append(brightness)
        if len(self.brightness_history) > self.history_size:
            self.brightness_history.pop(0)
        
        avg_brightness = np.mean(self.brightness_history)
        
        # Adjust threshold based on brightness
        # Low light (night): Lower threshold for better detection
        # Bright light (day): Higher threshold to reduce false positives
        if avg_brightness < 50:  # Very dark
            adjusted = self.min_threshold
        elif avg_brightness < 100:  # Dark
            adjusted = self.base_threshold - 0.05
        elif avg_brightness > 200:  # Very bright
            adjusted = self.base_threshold + 0.1
        elif avg_brightness > 150:  # Bright
            adjusted = self.base_threshold + 0.05
        else:  # Normal
            adjusted = self.base_threshold
        
        # Clamp to valid range
        adjusted = max(self.min_threshold, min(self.max_threshold, adjusted))
        
        return adjusted
    
    def reset(self):
        """Reset brightness history"""
        self.brightness_history = []


class CrowdDensityEstimator:
    """
    Estimate crowd density to adapt detection and tracking parameters
    """
    
    def __init__(self, frame_shape: Tuple[int, int]):
        self.frame_area = frame_shape[0] * frame_shape[1]
        self.density_history = []
        self.history_size = 30
    
    def estimate_density(self, num_detections: int, 
                        detection_boxes: Optional[list] = None) -> str:
        """
        Estimate crowd density level
        
        Args:
            num_detections: Number of people detected
            detection_boxes: List of bounding boxes (optional, for area calculation)
            
        Returns:
            Density level: 'low', 'medium', 'high', 'very_high'
        """
        # Calculate people per unit area
        density = num_detections / (self.frame_area / 1000000)  # Per megapixel
        
        self.density_history.append(density)
        if len(self.density_history) > self.history_size:
            self.density_history.pop(0)
        
        avg_density = np.mean(self.density_history)
        
        # Classify density
        if avg_density < 5:
            return 'low'
        elif avg_density < 15:
            return 'medium'
        elif avg_density < 30:
            return 'high'
        else:
            return 'very_high'
    
    def get_recommended_params(self, density: str) -> dict:
        """
        Get recommended detection parameters based on crowd density
        
        Args:
            density: Density level from estimate_density()
            
        Returns:
            Dictionary of recommended parameters
        """
        params = {
            'low': {
                'conf_threshold': 0.3,
                'iou_threshold': 0.45,
                'max_age': 60,
                'min_hits': 3
            },
            'medium': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.4,
                'max_age': 90,
                'min_hits': 2
            },
            'high': {
                'conf_threshold': 0.2,
                'iou_threshold': 0.35,
                'max_age': 120,
                'min_hits': 1
            },
            'very_high': {
                'conf_threshold': 0.15,
                'iou_threshold': 0.3,
                'max_age': 150,
                'min_hits': 1
            }
        }
        
        return params.get(density, params['medium'])
