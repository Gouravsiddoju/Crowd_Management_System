
import cv2
import argparse
import os
import glob
import sys
import numpy as np

def draw_instructions(image, state):
    """Draw overlay text instructions based on current state"""
    instructions = {
        'polygon': [
            "STEP 1: Draw Walkable Floor (Green)",
            "- Left Click: Add point",
            "- Right Click: Undo",
            "- 'd': Done",
            "- 'a': Auto-Detect"
        ],
        'exclusions': [
            "STEP 2: Draw Obstacles (Red)",
            "- Left Click: Add point",
            "- 'n': New obstacle", 
            "- 'd': Done"
        ],
        'front': [
            "STEP 3: Calibration - FRONT",
            "Draw box around a person CLOSE to camera",
            "- Click & Drag to draw",
            "- 'd': Confirm"
        ],
        'back': [
            "STEP 4: Calibration - BACK",
            "Draw box around a person FAR from camera",
            "- Click & Drag to draw",
            "- 'd': Finish"
        ]
    }
    
    lines = instructions.get(state, [])
    y = 30
    for line in lines:
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3) # Outline
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1) # Text
        y += 30

def click_event(event, x, y, flags, params):
    """Handle mouse clicks"""
    points = params['points']
    image = params['image']
    clone = params['clone']
    window_name = params['window_name']
    
    state = params['state'] # 'polygon', 'front', 'back'

    if event == cv2.EVENT_LBUTTONDOWN:
        if state == 'polygon' or state == 'exclusions':
            target_list = points if state == 'polygon' else params['current_exclusion']
            target_list.append([x, y])
            
            color = (0, 0, 255) if state == 'exclusions' else (0, 255, 0)
            
            # Draw point
            cv2.circle(image, (x, y), 5, color, -1)
            # Draw line if more than 1 point
            if len(target_list) > 1:
                cv2.line(image, tuple(target_list[-2]), tuple(target_list[-1]), color, 2)
            
            disp = image.copy()
            draw_instructions(disp, state)
            cv2.imshow(window_name, disp)
        
        elif state == 'front' or state == 'back':
            # Start drawing box
            params['dragging'] = True
            params['start_pt'] = (x, y)
            
            # No display update needed here really, wait for move

    elif event == cv2.EVENT_MOUSEMOVE:
        if (state == 'front' or state == 'back') and params['dragging']:
            img_copy = image.copy()
            cv2.rectangle(img_copy, params['start_pt'], (x, y), (0, 255, 255), 2)
            draw_instructions(img_copy, state)
            cv2.imshow(window_name, img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        if params['dragging']:
            params['dragging'] = False
            # Save box
            x1, y1 = params['start_pt']
            x2, y2 = x, y
            # Normalize coordinates
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            
            if state == 'front':
                params['front_box'] = box
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image, "FRONT REF", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif state == 'back':
                params['back_box'] = box
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(image, "BACK REF", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            disp = image.copy()
            draw_instructions(disp, state)
            cv2.imshow(window_name, disp)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Undo last point
        current_points = points if state == 'polygon' else params['current_exclusion']
        
        if (state == 'polygon' or state == 'exclusions') and current_points:
            current_points.pop()
            # Redraw
            image[:] = clone[:]
            
            # Draw main polygon if it exists
            if params['main_polygon']:
                pts = np.array(params['main_polygon'], dtype=np.int32)
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            
            # Draw completed exclusions
            for exc in params['exclusions']:
                pts = np.array(exc, dtype=np.int32)
                cv2.polylines(image, [pts], True, (0, 0, 255), 2)
                
            # Draw current points
            for i, pt in enumerate(current_points):
                color = (0, 0, 255) if state == 'exclusions' else (0, 255, 0)
                cv2.circle(image, tuple(pt), 5, color, -1)
                if i > 0:
                    cv2.line(image, tuple(current_points[i-1]), tuple(current_points[i]), color, 2)
            
            disp = image.copy()
            draw_instructions(disp, state)
            cv2.imshow(window_name, disp)

def process_video(video_path):
    """Process a single video to get boundary"""
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} not found.")
        return None

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame from {video_path}")
        return None

    # Resize for display if too large (but keep original coords for config)
    # Actually, better to show full size or specific scaled size, 
    # but for simplicity let's show exact frame to get exact coords.
    # If 4K, it might be too big for screen.
    
    scale = 1.0
    display_h = 900
    h, w = frame.shape[:2]
    
    if h > display_h:
        scale = display_h / h
        frame_display = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        frame_display = frame.copy()

    window_name = f"Define Boundary: {os.path.basename(video_path)}"
    clone = frame_display.copy()
    points = [] # Main polygon points
    
    params = {
        'points': points,
        'image': frame_display,
        'clone': clone,
        'window_name': window_name,
        'state': 'polygon',
        'dragging': False,
        'start_pt': (0,0),
        'front_box': None,
        'back_box': None,
        'main_polygon': [],
        'exclusions': [],
        'current_exclusion': []
    }
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event, params)
    
    print(f"\n--- {os.path.basename(video_path)} ---")
    print("See interactive window for instructions.")
    
    while True:
        disp = frame_display.copy()
        draw_instructions(disp, params['state'])
        cv2.imshow(window_name, disp)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
            
        elif key == ord('a'):
            if params['state'] == 'polygon' and not points:
                print("\nRunning Auto-Segmentation... Please wait.")
                try:
                    from auto_segmentation import SceneSegmenter
                    # Re-read original frame for best quality segmentation
                    # (frame_display might be resized)
                    # But for consistency with drawing coordinates, we should use what we see?
                    # No, segmenter needs good res. We can scale points back.
                    # Let's use frame_display for now to match current coordinate system easily.
                    
                    segmenter = SceneSegmenter()
                    inclusions, exclusions = segmenter.segment_frame(frame_display)
                    
                    if inclusions:
                        # Use the largest inclusion as main polygon
                        new_points = inclusions[0]
                        params['points'] = new_points
                        params['main_polygon'] = new_points
                        
                        # Add exclusions
                        params['exclusions'] = exclusions
                        
                        # Visualize
                        pts = np.array(new_points, dtype=np.int32)
                        cv2.polylines(frame_display, [pts], True, (0, 255, 0), 2)
                        
                        for exc in exclusions:
                            pts = np.array(exc, dtype=np.int32)
                            cv2.polylines(frame_display, [pts], True, (0, 0, 255), 2)
                            
                        print(f"  ✓ Found walkable floor and {len(exclusions)} obstacles.")
                        print("  Press 'd' to accept and continue, or 'c' to clear.")
                    else:
                        print("  ! No walkable floor detected.")
                        
                except Exception as e:
                    print(f"  ! Auto-detection failed: {e}")
                    print("  Make sure 'transformers' is installed.")
            
        elif key == ord('c'):
            # Clear current step
            if params['state'] == 'polygon':
                points.clear()
                params['main_polygon'] = []
            elif params['state'] == 'exclusions':
                params['current_exclusion'].clear()
                params['exclusions'].clear()
            
            frame_display[:] = clone[:]
            # Redraw logic needed if clearing partial state... simplified just reset image for now
            # Rerender existing valid parts
            if params['state'] != 'polygon' and params['main_polygon']:
                 pts = np.array(params['main_polygon'], dtype=np.int32)
                 cv2.polylines(frame_display, [pts], True, (0, 255, 0), 2)
            
            print(f"Reset current step: {params['state']}")
            
        elif key == ord('d'):
            if params['state'] == 'polygon':
                if len(points) > 2:
                    # Close polygon visually
                    cv2.line(frame_display, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
                    params['main_polygon'] = list(points)
                    params['state'] = 'exclusions'
                    print("\nSTEP 2: Draw Exclusion Zones (Obstacles) - Red")
                    print("  Left Click  : Add point to CURRENT obstacle")
                    print("  'n'         : Finish CURRENT obstacle and start NEW one")
                    print("  'd'         : Done with ALL obstacles -> Next Step")
            
            elif params['state'] == 'exclusions':
                # Save any pending exclusion polygon if valid
                if len(params['current_exclusion']) > 2:
                    params['exclusions'].append(list(params['current_exclusion']))
                    pts = np.array(params['current_exclusion'], dtype=np.int32)
                    cv2.polylines(frame_display, [pts], True, (0, 0, 255), 2)
                    params['current_exclusion'].clear()

                params['state'] = 'front'
                print("\nSTEP 3: Draw Box around a Person at the FRONT (Closest)")
                print("  Click & Drag : Draw box")
                print("  'd'          : Confirm -> Next Step")
            
            elif params['state'] == 'front':
                if params['front_box']:
                    params['state'] = 'back'
                    print("\nSTEP 4: Draw Box around a Person at the BACK (Furthest)")
                    print("  Click & Drag : Draw box")
                    print("  'd'          : Finish")
                    
            elif params['state'] == 'back':
                if params['back_box']:
                    break # Finished all steps
        
        elif key == ord('n'):
            if params['state'] == 'exclusions':
                 if len(params['current_exclusion']) > 2:
                    # Finish current poly
                    params['exclusions'].append(list(params['current_exclusion']))
                    pts = np.array(params['current_exclusion'], dtype=np.int32)
                    cv2.polylines(frame_display, [pts], True, (0, 0, 255), 2)
                    params['current_exclusion'] = []
                    print("  Started new obstacle definition...")
                 else:
                     print("  Need at least 3 points for an obstacle.")

    cv2.destroyWindow(window_name)
    
    # Scale points back to original resolution
    real_data = {'points': [], 'exclusions': [], 'front': [], 'back': []}
    
    for pt in params['main_polygon']:
        real_data['points'].append([int(pt[0] / scale), int(pt[1] / scale)])
        
    for poly in params['exclusions']:
        scaled_poly = [[int(pt[0] / scale), int(pt[1] / scale)] for pt in poly]
        real_data['exclusions'].append(scaled_poly)
        
    if params['front_box']:
        fb = params['front_box']
        real_data['front'] = [int(p / scale) for p in fb]
        
    if params['back_box']:
        bb = params['back_box']
        real_data['back'] = [int(p / scale) for p in bb]
        
    return real_data

def main():
    # ==========================================
    # INPUT CONFIGURATION
    # Paste your video file path or folder path here:
    INPUT_SOURCE = "clip/2.wmv" 
    # Examples:
    # INPUT_SOURCE = "clip"                   # Process all videos in 'clip' folder
    # INPUT_SOURCE = "/path/to/my/video.mp4"  # Process specific video
    # ==========================================

    parser = argparse.ArgumentParser(description='Interactive Boundary Selection Tool')
    parser.add_argument('--input', type=str, help='Video file or folder containing videos')
    args = parser.parse_args()

    # Priority: Command Line > INPUT_SOURCE variable
    source_to_use = INPUT_SOURCE
    if args.input:
        source_to_use = args.input

    video_paths = []
    
    if os.path.isdir(source_to_use):
        exts = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
        for ext in exts:
            video_paths.extend(glob.glob(os.path.join(source_to_use, ext)))
        video_paths.sort()
    elif os.path.exists(source_to_use):
        video_paths = [source_to_use]
    else:
        print(f"Error: Input source '{source_to_use}' not found!")
        return

    if not video_paths:
        print(f"No videos found in {source_to_use}!")
        return

    print("="*60)
    print("INTERACTIVE BOUNDARY SETUP")
    print("="*60)

    configs = {}

    for video_path in video_paths:
        points = process_video(video_path)
        if points and len(points) >= 3:
            name = os.path.basename(video_path)
            configs[name] = points
            print(f"Captured {len(points)} points for {name}")
            
            # Ask for Area
            try:
                area_str = input(f"Enter Total Physical Area in m^2 for {name} (e.g. 50.0) [Enter to skip]: ")
                if area_str.strip():
                    points['area_sq_meters'] = float(area_str.strip())
            except ValueError:
                print("Invalid area value entered. Skipping area.")

    print("="*60)
    
    # Ask to save
    if configs:
        save = input("\nDo you want to update config.yaml with these new boundaries? (y/n): ")
        if save.lower() == 'y':
            try:
                import yaml
                import shutil
                
                config_path = 'config.yaml'
                
                # Check if exists
                if not os.path.exists(config_path):
                    print("Error: config.yaml not found in current directory.")
                    return

                # Backup
                shutil.copy(config_path, config_path + '.bak')
                print(f"Backed up config to {config_path}.bak")
                
                # Read existing
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                if 'boundaries' not in config_data:
                    config_data['boundaries'] = {'enabled': True, 'default_type': 'inclusion', 'cameras': {}}
                
                if 'cameras' not in config_data['boundaries']:
                    config_data['boundaries']['cameras'] = {}
                    
                # Update cameras
                for name, data in configs.items():
                    # Calculate calibration
                    f_box = data['front']
                    b_box = data['back']
                    
                    if not f_box or not b_box:
                        print(f"Skipping {name}: Missing calibration boxes")
                        continue
                        
                    f_area = (f_box[2]-f_box[0]) * (f_box[3]-f_box[1])
                    f_y = (f_box[1] + f_box[3]) // 2
                    
                    b_area = (b_box[2]-b_box[0]) * (b_box[3]-b_box[1])
                    b_y = (b_box[1] + b_box[3]) // 2
                    
                    config_data['boundaries']['cameras'][name] = {
                        'type': 'inclusion',
                        'polygon': data['points'],
                        'calibration': {
                            'front_y': int(f_y),
                            'front_area': int(f_area),
                            'back_y': int(b_y),
                            'back_area': int(b_area)
                        },
                        'exclusions': data.get('exclusions', [])
                    }
                    
                    if 'area_sq_meters' in data:
                        config_data['boundaries']['cameras'][name]['area_sq_meters'] = data['area_sq_meters']
                    
                # Write back
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f, sort_keys=False, default_flow_style=None)
                    
                print(f"Successfully updated {config_path}!")
                print("You can verify the changes by opening the file.")
                
            except ImportError:
                print("Error: PyYAML not installed. Run 'pip install pyyaml'")
                # Fallback to printing
                print_manual_config(configs)
            except Exception as e:
                print(f"Error saving config: {e}")
        else:
            print_manual_config(configs)

def print_manual_config(configs):
    print("\n" + "="*60)
    print("MANUAL CONFIGURATION (Copy to config.yaml)")
    print("="*60)
    for name, data in configs.items():
        print(f"    \"{name}\":")
        # ... (print logic similar to before if needed for backup)
        print(f"      polygon: {data['points']}")
        if data.get('exclusions'):
            print(f"      exclusions: {data['exclusions']}")
    print("="*60)

if __name__ == "__main__":
    main()
