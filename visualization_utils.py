import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import math
import os

class DepthMapAnimator:
    """
    Creates an animated video of the internal depth map with moving people.
    """
    def __init__(self, boundary_config, output_path, fps=30, resolution=(1280, 720)):
        self.boundary_config = boundary_config
        self.output_path = output_path
        self.width, self.height = resolution
        self.fps = fps
        self.writer = None
        
        # Pre-render background
        self.background = self._create_background()
        
        # Initialize writer
        self._init_writer()

    def _create_background(self):
        """Create the static background with zones"""
        bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255 # White background
        
        # Draw Polygon (Green)
        if 'polygon' in self.boundary_config:
            pts = np.array(self.boundary_config['polygon'], dtype=np.int32)
            # Fill light green
            overlay = bg.copy()
            cv2.fillPoly(overlay, [pts], (220, 255, 220)) # BGR
            cv2.addWeighted(overlay, 0.5, bg, 0.5, 0, bg)
            # Outline
            cv2.polylines(bg, [pts], True, (0, 200, 0), 2)
            
        # Draw Exclusions (Red)
        if 'exclusions' in self.boundary_config:
            for exc in self.boundary_config['exclusions']:
                pts = np.array(exc, dtype=np.int32)
                # Fill light red
                overlay = bg.copy()
                cv2.fillPoly(overlay, [pts], (220, 220, 255))
                cv2.addWeighted(overlay, 0.5, bg, 0.5, 0, bg)
                # Outline
                cv2.polylines(bg, [pts], True, (0, 0, 255), 2)
                
        # Draw Ref Lines
        calib = self.boundary_config.get('calibration', {})
        if calib:
            fy = calib.get('front_y', 0)
            by = calib.get('back_y', 0)
            if fy: cv2.line(bg, (0, fy), (self.width, fy), (150, 150, 150), 1, cv2.LINE_AA)
            if by: cv2.line(bg, (0, by), (self.width, by), (150, 150, 150), 1, cv2.LINE_AA)
            
            # Add Labels
            cv2.putText(bg, "Back Ref", (10, by-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
            cv2.putText(bg, "Front Ref", (10, fy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

        cv2.putText(bg, "Internal Depth Map Tracking", (self.width//2 - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        return bg

    def _init_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        print(f"✓ Initialized depth map video: {self.output_path}")

    def update(self, tracks):
        """
        Update the animation with current tracks.
        tracks: list of Track objects or dict of simple (id, box)
        """
        frame = self.background.copy()
        
        calib = self.boundary_config.get('calibration', {})
        fy, fa = calib.get('front_y', 0), calib.get('front_area', 0)
        by, ba = calib.get('back_y', 0), calib.get('back_area', 0)
        
        # Draw tracked people
        for track in tracks:
            # Handle different track object types
            if hasattr(track, 'bbox'): # Track object
                x1, y1, x2, y2 = map(int, track.bbox)
                tid = track.track_id
            else: # Fallback assuming tuple (box, id) or similar, strictly adhering to TrackManager output
                 # If it's the list of Track objects from run_multicam:
                 # It is indeed Track objects.
                 continue

            cx = (x1 + x2) // 2
            cy = y2 # Feet position
            
            # Calculate dynamic radius based on depth (Y)
            radius = 15 # Default
            if fy != by:
                y_norm = (cy - by) / (fy - by)
                pixel_area = ba + (fa - ba) * y_norm
                pixel_area = max(pixel_area, 100)
                radius = int(math.sqrt(pixel_area / math.pi) / 2) # Half size for cleaner look? 
                # User image showed overlapping circles approx personal space size.
                # Let's use radius ~ sqrt(Area/pi)
                radius = int(math.sqrt(pixel_area / math.pi))
                
            # Draw Circle
            # Blue with alpha
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), radius, (255, 0, 0), -1) # BGR: Blue
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Draw Border
            cv2.circle(frame, (cx, cy), radius, (200, 0, 0), 1)
            
            # Draw ID
            cv2.putText(frame, str(tid), (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()
            print(f"✓ Saved animated depth map: {self.output_path}")

def generate_depth_map(boundary_config, output_path, resolution=(1280, 720)):
    """
    Generates and saves a visualization of the system's internal depth/capacity map.
    Mimics the user's "blue circle packing" visualization.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    if 'polygon' not in boundary_config:
        return

    polygon = boundary_config['polygon']
    exclusions = boundary_config.get('exclusions', [])
    calib = boundary_config.get('calibration', {})
    
    if not calib:
        return

    width, height = resolution
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("System's Internal 3D Depth Map (Visualization)")
    ax.set_xlabel("X (Pixels)")
    ax.set_ylabel("Y (Pixels - Depth)")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0) # Invert Y for image coords
    ax.grid(alpha=0.3, linestyle=':')
    
    # 1. Draw Inclusions (Green)
    poly_patch = patches.Polygon(polygon, closed=True, linewidth=2, edgecolor='green', facecolor=(0, 1, 0, 0.1), label='Walkable Floor')
    ax.add_patch(poly_patch)
    
    # 2. Draw Exclusions (Red)
    for exc in exclusions:
        exc_patch = patches.Polygon(exc, closed=True, linewidth=2, edgecolor='red', facecolor=(1, 0, 0, 0.2), label='Obstacle/Exclusion')
        ax.add_patch(exc_patch)
        
    # 3. Draw Reference Lines
    front_y = calib['front_y']
    back_y = calib['back_y']
    ax.axhline(y=front_y, color='gray', linestyle='--', label=f'Front Ref (Y={front_y})')
    ax.axhline(y=back_y, color='gray', linestyle='--', label=f'Back Ref (Y={back_y})')
    
    # 4. Generate "Heatmap" / Packing Visualization
    # We want to fill the polygon with circles representing person size at that depth
    
    # Create mask for valid area
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    for exc in exclusions:
        cv2.fillPoly(mask, [np.array(exc, dtype=np.int32)], 0)
        
    # Calibration params
    fy, fa = front_y, calib['front_area']
    by, ba = back_y, calib['back_area']
    
    # Sampling step (don't check every pixel, too slow)
    step_y = 20
    step_x = 20
    
    circles = []
    
    # Get all valid points
    ys, xs = np.where(mask > 0)
    
    # We want to pack circles. A simple robust way for visualization:
    # Iterate Y from back to front, place circles along X if space is free.
    # Since this is just a visualization, we can use a randomized or grid approach.
    # Let's use a grid approach with jitter for "natural" look.
    
    rng = np.random.RandomState(42)
    
    # Optimization: calculate size for ALL Ys first
    # Size = sqrt(Area / pi) * scale? Or just Area as circle area?
    # User image circles look like they represent the "personal space" or bounding box area.
    # Bounding box area = w * h. Circle radius r = sqrt(w*h/pi).
    
    # Normalize Y
    y_range_len = max(1, fy - by)
    
    # We will try to place circles in a grid
    occupied = np.zeros((height, width), dtype=bool)
    
    # Iterate through potentially valid points (grid)
    # To look like the user image, we want overlapping semi-transparent circles
    # spread across the whole area to show the "field".
    
    points_to_plot = []
    
    for y in range(0, height, 30): # Step size 30
        for x in range(0, width, 30):
            if mask[y, x] > 0:
                # Calculate size at this depth
                # Linear interpolation of AREA
                y_norm = (y - by) / (fy - by)
                pixel_area = ba + (fa - ba) * y_norm
                pixel_area = max(pixel_area, 100) # Min size
                
                # Convert area to radius (approximate)
                radius = math.sqrt(pixel_area / math.pi)
                
                points_to_plot.append((x, y, radius))
    
    # Plot circles
    # Using a Collection is faster than individual patches
    from matplotlib.collections import PatchCollection
    
    patches_list = []
    for (x, y, r) in points_to_plot:
        circle = patches.Circle((x, y), radius=r)
        patches_list.append(circle)
        
    p = PatchCollection(patches_list, alpha=0.15, facecolor='blue', edgecolor='none')
    ax.add_collection(p)
    
    # Legend
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"✓ Saved depth map visualization: {output_path}")

