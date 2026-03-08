"""
Main Deployment Script for Robust Crowd Counting System
Optimized for Indian railway stations and bus stands

Usage:
    python run_station_counter.py --config config.yaml
    python run_station_counter.py --stream rtsp://192.168.1.100:554/stream
    python run_station_counter.py --input video.mp4 --mode ACCURATE
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import yaml
import argparse
import time
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from preprocessing import FramePreprocessor, AdaptiveThresholder, CrowdDensityEstimator
from stream_handler import StreamHandler, OutputHandler
from advanced_tracker import TrackManager
from zone_counter import ZoneCounter, Zone
from analytics import AnalyticsEngine, PerformanceProfiler


class StationCounter:
    """Main crowd counting system for railway/bus stations"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        print("=" * 70)
        print("ROBUST CROWD COUNTING SYSTEM")
        print("Optimized for Railway Stations & Bus Stands")
        print("=" * 70 + "\n")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Apply performance mode overrides
        self._apply_performance_mode()
        
        # Initialize components (except stream which may be overridden)
        self._init_model()
        self._init_tracking()
        self._init_zones()
        self._init_visualization()
        
        # State
        self.frame_count = 0
        self.running = False
        self.profiler = PerformanceProfiler() if self.config['advanced']['profile_performance'] else None
        
        print("\n✓ System initialized (awaiting stream setup)")
        print("Press 'q' to quit, 's' to save screenshot, 'p' to pause\n")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Loaded configuration: {config_path}")
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration if file not found"""
        return {
            'model': {'name': 'yolov8x.pt', 'imgsz': 1280, 'conf_threshold': 0.25, 'device': 'cuda'},
            'tracking': {'tracker': 'botsort.yaml', 'reid_enabled': True, 'reid_threshold': 0.6, 'max_age': 90},
            'preprocessing': {'enabled': True, 'lighting_normalization': True, 'denoising': True},
            'stream': {'type': 'file', 'file_path': 'cr.mp4'},
            'output': {'display_preview': True, 'save_video': False},
            'analytics': {'enabled': True, 'log_to_console': True, 'log_to_file': True},
            'counting': {'mode': 'occupancy', 'zones_enabled': False},
            'performance': {'mode': 'BALANCED', 'frame_skip': 1},
            'advanced': {'debug_mode': False, 'profile_performance': False}
        }
    
    def _apply_performance_mode(self):
        """Apply performance mode overrides"""
        mode = self.config['performance']['mode']
        
        if 'performance_modes' in self.config and mode in self.config['performance_modes']:
            overrides = self.config['performance_modes'][mode]
            print(f"✓ Applying performance mode: {mode}")
            
            for key, value in overrides.items():
                if key == 'model_name':
                    self.config['model']['name'] = value
                elif key in ['imgsz', 'conf_threshold']:
                    self.config['model'][key] = value
                elif key == 'frame_skip':
                    self.config['performance']['frame_skip'] = value
            
            print(f"  Model: {self.config['model']['name']}, Size: {self.config['model']['imgsz']}")
    
    def _init_model(self):
        """Initialize YOLO model"""
        model_name = self.config['model']['name']
        device = self.config['model']['device']
        
        # Check CUDA availability
        if str(device) in ['cuda', '0'] and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            print("Warning: CUDA not available, using CPU")
            self.device = 'cpu'
            self.config['model']['device'] = 'cpu'
        
        print(f"Loading model: {model_name} on {self.device}...")
        self.model = YOLO(model_name)
        
        print(f"✓ Model loaded: {model_name}")
        if device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {gpu_name}")
    
    def _init_stream(self):
        """Initialize video stream"""
        # Close old stream if exists
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.release()
            
        self.stream = StreamHandler(self.config['stream'])
        
        if not self.stream.is_connected():
            print("Error: Could not connect to video stream")
            sys.exit(1)
        
        # Initialize output handler
        stream_props = self.stream.get_properties()
        self.output = OutputHandler(self.config['output'], stream_props)
        
        self.fps = stream_props['fps']
        self.frame_width = stream_props['width']
        self.frame_height = stream_props['height']
    
    def _init_tracking(self):
        """Initialize tracking system"""
        self.track_manager = TrackManager(self.config['tracking'])
        print(f"✓ Tracking initialized (Re-ID: {'Enabled' if self.config['tracking']['reid_enabled'] else 'Disabled'})")
    
    def _init_preprocessing(self):
        """Initialize preprocessing"""
        self.preprocessor = FramePreprocessor(self.config['preprocessing'], device=self.device)
        self.adaptive_threshold = AdaptiveThresholder(
            base_threshold=self.config['model']['conf_threshold'],
            min_threshold=0.15,
            max_threshold=0.5
        )
        self.density_estimator = CrowdDensityEstimator((self.frame_height, self.frame_width))
        
        if self.config['preprocessing']['enabled']:
            print("✓ Preprocessing enabled")
    
    def _init_zones(self):
        """Initialize zone counting (if enabled)"""
        self.zone_counter = None
        
        if self.config['counting']['zones_enabled'] and self.config['counting']['zones']:
            zones = []
            for zone_config in self.config['counting']['zones']:
                zone = Zone(
                    name=zone_config['name'],
                    polygon=zone_config['polygon'],
                    entry_line=tuple(zone_config.get('entry_line', [])) if zone_config.get('entry_line') else None,
                    exit_line=tuple(zone_config.get('exit_line', [])) if zone_config.get('exit_line') else None,
                    alert_threshold=zone_config.get('alert_threshold', 0)
                )
                zones.append(zone)
            
            self.zone_counter = ZoneCounter(zones, fps=self.fps)
            print(f"✓ Zone counting enabled ({len(zones)} zones)")
        else:
            print("  Zone counting disabled")
    
    def _init_analytics(self):
        """Initialize analytics"""
        self.analytics = AnalyticsEngine(self.config['analytics'], fps=self.fps)
        
        if self.analytics.enabled:
            print(f"✓ Analytics enabled (logging to: {self.analytics.log_file})")
    
    def _init_visualization(self):
        """Initialize visualization settings"""
        self.show_preview = self.config['output']['display_preview']
        
        if self.show_preview:
            cv2.namedWindow('Crowd Counting System', cv2.WINDOW_NORMAL)
            preview_w = self.config['output'].get('preview_width', 1280)
            preview_h = self.config['output'].get('preview_height', 720)
            cv2.resizeWindow('Crowd Counting System', preview_w, preview_h)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        
        # Profiling
        if self.profiler:
            self.profiler.start_stage('preprocessing')
        
        # 1. Preprocessing
        processed_frame = self.preprocessor.process(frame)
        
        # 2. Adaptive threshold
        conf_threshold = self.adaptive_threshold.get_adaptive_threshold(processed_frame)
        
        if self.profiler:
            self.profiler.end_stage()
            self.profiler.start_stage('detection')
        
        # 3. Detection & Tracking
        results = self.model.track(
            processed_frame,
            persist=True,
            device=self.device,
            verbose=False,
            tracker=self.config['tracking']['tracker'],
            imgsz=self.config['model']['imgsz'],
            conf=conf_threshold,
            iou=self.config['model'].get('iou_threshold', 0.45)
        )
        
        if self.profiler:
            self.profiler.end_stage()
            self.profiler.start_stage('tracking_update')
        
        # 4. Extract detections and update tracking
        yolo_tracks = {}
        current_count = 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if int(box.cls[0]) == 0:  # Person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Apply filtering
                    if not self._is_valid_detection(x1, y1, x2, y2):
                        continue
                    
                    current_count += 1
                    
                    # Get track ID
                    if box.id is not None:
                        track_id = int(box.id[0])
                        yolo_tracks[track_id] = ((x1, y1, x2, y2), conf)
        
        # Update track manager
        active_tracks = self.track_manager.update([], frame, self.frame_count, yolo_tracks)
        
        if self.profiler:
            self.profiler.end_stage()
            self.profiler.start_stage('zone_update')
        
        # 5. Update zones (if enabled)
        if self.zone_counter:
            zone_tracks = {}
            for track_id, (bbox, conf) in yolo_tracks.items():
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                zone_tracks[track_id] = (cx, cy, bbox)
            
            self.zone_counter.update(zone_tracks)
        
        if self.profiler:
            self.profiler.end_stage()
            self.profiler.start_stage('visualization')
        
        # 6. Visualization
        vis_frame = frame.copy()
        
        # Draw detections
        for track_id, (bbox, conf) in yolo_tracks.items():
            x1, y1, x2, y2 = bbox
            
            # Draw box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), 
                         self.config['output']['box_color'], 
                         self.config['output']['box_thickness'])
            
            # Draw ID
            if self.config['output']['show_track_ids']:
                label = f"ID:{track_id}"
                cv2.putText(vis_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           self.config['output']['text_size'],
                           self.config['output']['text_color'], 2)
        
        # Draw zones
        if self.zone_counter:
            self.zone_counter.draw(vis_frame)
            if self.config['counting'].get('show_zone_stats', False):
                self.zone_counter.draw_statistics_panel(vis_frame)
        
        # Draw HUD
        self.analytics.draw_hud(vis_frame, current_count, len(self.track_manager.unique_ids))
        
        if self.profiler:
            self.profiler.end_stage()
        
        # 7. Update analytics
        # Filter for newly completed tracks (ended at current frame)
        newly_completed = [t.duration_seconds(self.fps) for t in self.track_manager.completed_tracks 
                          if t.last_seen == self.frame_count]
                          
        metrics = {
            'current_count': current_count,
            'unique_count': len(self.track_manager.unique_ids),
            'frame_time': 0,  # Will be updated below
            'completed_dwell_times': newly_completed
        }
        
        return vis_frame, metrics
    
    def _is_valid_detection(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Filter false positives"""
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / width if width > 0 else 0
        
        filtering = self.config.get('filtering', {})
        min_height = filtering.get('min_person_height', 80)
        min_width = filtering.get('min_person_width', 35)
        min_ratio = filtering.get('min_aspect_ratio', 1.2)
        max_ratio = filtering.get('max_aspect_ratio', 5.0)
        
        return (height >= min_height and 
                width >= min_width and 
                min_ratio <= aspect_ratio <= max_ratio)
    
    def run(self):
        """Main processing loop"""
        self.running = True
        frame_skip = self.config['performance']['frame_skip']
        paused = False
        
        print("\n" + "=" * 70)
        print("STARTING PROCESSING")
        print("=" * 70 + "\n")
        
        frame_time_start = time.time()
        
        try:
            while self.running:
                # Handle pause
                if paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('p'):  # Unpause
                        paused = False
                        print("▶  Resumed")
                    elif key == ord('q'):  # Quit
                        break
                    continue
                
                # Read frame
                ret, frame = self.stream.read()
                
                if not ret:
                    print("\nEnd of stream")
                    break
                
                self.frame_count += 1
                
                # Frame skipping
                if frame_skip > 1 and self.frame_count % frame_skip != 0:
                    continue
                
                # Process frame
                frame_start = time.time()
                vis_frame, metrics = self.process_frame(frame)
                frame_elapsed = time.time() - frame_start
                
                metrics['frame_time'] = frame_elapsed
                
                # Update analytics
                self.analytics.update(metrics)
                
                # Save output
                self.output.write(vis_frame, self.frame_count)
                
                # Show preview
                if self.show_preview:
                    cv2.imshow('Crowd Counting System', vis_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # Quit
                        print("\nStopping...")
                        break
                    elif key == ord('s'):  # Screenshot
                        screenshot_path = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(screenshot_path, vis_frame)
                        print(f"✓ Screenshot saved: {screenshot_path}")
                    elif key == ord('p'):  # Pause
                        paused = True
                        print("⏸  Paused (press 'p' to resume)")
                
                # Progress indicator (every 30 frames)
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - frame_time_start
                    fps = 30 / elapsed
                    frame_time_start = time.time()
                    
                    print(f"Frame {self.frame_count}: {metrics['current_count']} people, "
                          f"{fps:.1f} FPS, Unique: {metrics['unique_count']}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n" + "=" * 70)
        print("SHUTTING DOWN")
        print("=" * 70)
        
        # Release stream and output
        self.stream.release()
        self.output.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Print analytics summary
        if self.analytics:
            self.analytics.close()
        
        # Print tracking stats
        track_stats = self.track_manager.get_statistics()
        print(f"\nTracking Statistics:")
        print(f"  Total unique people: {track_stats['unique_ids']}")
        print(f"  Re-ID matches: {track_stats['reid_matches']}")
        print(f"  Re-ID success rate: {track_stats['reid_success_rate']:.1f}%")
        
        # Print zone stats
        if self.zone_counter:
            zone_stats = self.zone_counter.get_statistics()
            print(f"\nZone Statistics:")
            for zone_name, stats in zone_stats['zones'].items():
                print(f"  {zone_name}:")
                print(f"    Occupancy: {stats['current_count']} (peak: {stats['peak_count']})")
                if 'entries' in stats:
                    print(f"    Entries: {stats['entries']}, Exits: {stats.get('exits', 0)}")
        
        # Print performance profile
        if self.profiler:
            self.profiler.print_report()
        
        print("\n✓ Shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Robust Crowd Counting for Railway/Bus Stations')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input', type=str,
                       help='Input video file (overrides config)')
    parser.add_argument('--stream', type=str,
                       help='RTSP stream URL (overrides config)')
    parser.add_argument('--mode', type=str, choices=['FAST', 'BALANCED', 'ACCURATE'],
                       help='Performance mode (overrides config)')
    parser.add_argument('--output', type=str,
                       help='Output video path (enables video saving)')
    
    args = parser.parse_args()
    
    # Create system
    system = StationCounter(config_path=args.config)
    
    # Apply command-line overrides
    if args.input:
        system.config['stream']['type'] = 'file'
        system.config['stream']['file_path'] = args.input
    
    if args.stream:
        system.config['stream']['type'] = 'rtsp'
        system.config['stream']['rtsp_url'] = args.stream
        
    # Now initialize stream and stream-dependent components
    system._init_stream()
    system._init_preprocessing()
    system._init_analytics()
    
    if args.mode:
        system.config['performance']['mode'] = args.mode
        system._apply_performance_mode()
        system._init_model()
    
    if args.output:
        system.config['output']['save_video'] = True
        system.config['output']['video_path'] = args.output
    
    # Run system
    system.run()


if __name__ == '__main__':
    main()
