"""
Analytics and Logging Module
Tracks metrics, generates logs, and provides alerts
"""

import csv
import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class AnalyticsEngine:
    """Comprehensive analytics for crowd counting system"""
    
    def __init__(self, config: dict, fps: int = 30):
        self.config = config
        self.fps = fps
        self.enabled = config.get('enabled', True)
        
        # Logging settings
        self.log_to_console = config.get('log_to_console', True)
        self.log_to_file = config.get('log_to_file', True)
        # Allow overriding log_file from config or use default
        self.log_file = config.get('log_file_path', config.get('log_file', 'analytics.csv'))
        self.json_file = config.get('json_file_path', None) # Path to write live stats JSON
        self.log_interval = config.get('log_interval', 1)  # seconds
        
        # Metrics tracking
        self.track_occupancy = config.get('track_occupancy', True)
        self.track_peak = config.get('track_peak_occupancy', True)
        self.track_flow = config.get('track_flow_rate', True)
        self.track_dwell = config.get('track_dwell_time', True)
        self.track_fps_metric = config.get('track_fps', True)
        
        # Alert settings
        self.alerts_enabled = config.get('alerts_enabled', False)
        self.alert_threshold = config.get('alert_threshold', 100)
        self.alert_sound = config.get('alert_sound', False)
        self.alert_webhook = config.get('alert_webhook', '')
        
        # Data storage
        self.occupancy_history = deque(maxlen=1800)  # 1 minute at 30fps
        self.unique_history = deque(maxlen=1800)     # Track unique count history for flow rate
        self.peak_occupancy = 0
        self.flow_history = deque(maxlen=1800)
        self.fps_history = deque(maxlen=90)  # 3 seconds at 30fps
        self.dwell_times = []
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.frame_times = deque(maxlen=30)
        
        # Session stats
        self.total_unique_people = 0
        self.alert_count = 0
        self.last_alert_time = 0
        
        
        # Physical parameters
        self.phys_params = config.get('physical_params', {})
        self.base_footprint = self.phys_params.get('base_footprint_m2', 0.4)
        self.buffer_high = self.phys_params.get('buffer_factor_high', 2.0)
        self.buffer_low = self.phys_params.get('buffer_factor_low', 1.2)
        
        self.utilization_smoother = deque(maxlen=30)
        
        # Initialize CSV log
        if self.log_to_file:
            self._init_csv_log()
    
    def _init_csv_log(self):
        """Initialize CSV log file with headers"""
        try:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'datetime',
                    'current_occupancy',
                    'peak_occupancy',
                    'unique_people',
                    'utilization_pct',
                    'est_capacity',
                    'capacity_status',
                    'flow_rate_per_min',
                    'avg_dwell_time_sec',
                    'processing_fps',
                    'processing_fps',
                    'alert_active',
                    'area_used_sqm',
                    'area_free_sqm'
                ])
            print(f"✓ Analytics log initialized: {self.log_file}")
        except Exception as e:
            print(f"Warning: Could not initialize CSV log: {e}")
            self.log_to_file = False
    
    def update(self, metrics: dict):
        """
        Update analytics with new metrics
        
        Args:
            metrics: Dict containing current metrics:
                - current_count: Current people count
                - unique_count: Total unique people seen
                - zone_stats: Optional zone statistics
                - frame_time: Time taken to process frame
        """
        if not self.enabled:
            return
        
        current_time = time.time()
        current_count = metrics.get('current_count', 0)
        
        # Update occupancy
        if self.track_occupancy:
            self.occupancy_history.append(current_count)
            if current_count > self.peak_occupancy:
                self.peak_occupancy = current_count
        
        # Update unique people
        self.total_unique_people = metrics.get('unique_count', 0)
        self.unique_history.append(self.total_unique_people)
        
        # Update FPS
        if self.track_fps_metric:
            frame_time = metrics.get('frame_time', 0)
            if frame_time > 0:
                self.frame_times.append(frame_time)
                current_fps = 1.0 / np.mean(self.frame_times)
                self.fps_history.append(current_fps)
        
        if 'completed_dwell_times' in metrics:
            self.dwell_times.extend(metrics['completed_dwell_times'])
        
        # Check alerts
        alert_active = False
        if self.alerts_enabled and current_count >= self.alert_threshold:
            alert_active = True
            # Trigger alert if cooldown period passed (60 seconds)
            if current_time - self.last_alert_time > 60:
                self._trigger_alert(current_count)
                self.last_alert_time = current_time
                self.alert_count += 1
        
    
        
        # Log to file periodically
        if self.log_to_file and (current_time - self.last_log_time) >= self.log_interval:
            polygon_pixels = metrics.get('polygon_pixels', 0)
            occupied_pixels = metrics.get('occupied_pixels', 0)
            capacity = metrics.get('capacity', 0)
            area_sq_meters = metrics.get('area_sq_meters', 0)
            self._log_to_csv(current_count, alert_active, polygon_pixels, occupied_pixels, capacity, area_sq_meters)
            if self.json_file:
                self._log_to_json(current_count, alert_active, capacity, area_sq_meters)
            self.last_log_time = current_time
    
    def _log_to_csv(self, current_count: int, alert_active: bool, polygon_pixels: float = 0, occupied_pixels: float = 0, metrics_capacity: int = 0, area_sq_meters: float = 0):
        """Log current metrics to CSV file"""
        try:
            timestamp = time.time()
            dt = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate metrics
            flow_rate = self._calculate_flow_rate()
            avg_dwell = np.mean(self.dwell_times) if self.dwell_times else 0
            avg_fps = np.mean(self.fps_history) if self.fps_history else 0
            
            # Static Capacity (calculated at startup and passed here)
            est_capacity = metrics_capacity if metrics_capacity > 0 else 0
            
            # Geometric Footprint Model for Utilization
            # Dynamic Buffering: 2.0 (Comfortable) -> 1.2 (Crowded)
            # Linearly interpolate buffer based on fullness (0% -> high buffer, 100% -> low buffer)
            # Simple approach: Scale buffer based on count vs capacity
            
            dynamic_buffer = self.buffer_high
            if est_capacity > 0:
                fullness_ratio = min(1.0, current_count / est_capacity)
                # Linear interpolation: high -> low
                dynamic_buffer = self.buffer_high - (fullness_ratio * (self.buffer_high - self.buffer_low))
            
            # Calculate Used Area
            used_area = current_count * (self.base_footprint * dynamic_buffer)
            
            # Calculate Utilization %
            raw_utilization_pct = 0.0
            if area_sq_meters > 0:
                raw_utilization_pct = (used_area / area_sq_meters) * 100
            elif est_capacity > 0:
                 # Fallback if area not passed but capacity is
                 raw_utilization_pct = (current_count / est_capacity) * 100

            # Apply Temporal Smoothing (30-frame rolling average)
            self.utilization_smoother.append(raw_utilization_pct)
            utilization_pct = np.mean(self.utilization_smoother)
            
            # Status based on utilization
            capacity_status = "NORMAL"
            if utilization_pct > 50:
                capacity_status = "HIGH"
            if utilization_pct > 80:
                capacity_status = "CRITICAL"

            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{timestamp:.2f}",
                    dt,
                    current_count,
                    self.peak_occupancy,
                    self.total_unique_people,
                    f"{utilization_pct:.1f}%",
                    est_capacity,
                    capacity_status,
                    f"{flow_rate:.1f}",
                    f"{avg_dwell:.1f}",
                    f"{avg_fps:.1f}",
                    f"{avg_fps:.1f}",
                    alert_active,
                    f"{used_area:.2f}",
                    f"{max(0, area_sq_meters - used_area):.2f}"
                ])
        except Exception as e:
            print(f"Warning: Could not log to CSV: {e}")

    def _log_to_json(self, current_count: int, alert_active: bool, metrics_capacity: int = 0, area_sq_meters: float = 0):
        """Write current stats to JSON for frontend"""
        try:
            import json
            import os
            
            # Calculate metrics
            est_capacity = metrics_capacity if metrics_capacity > 0 else 0
            
            # Utilization calc
            dynamic_buffer = self.buffer_high
            if est_capacity > 0:
                fullness_ratio = min(1.0, current_count / est_capacity)
                dynamic_buffer = self.buffer_high - (fullness_ratio * (self.buffer_high - self.buffer_low))
            
            used_area = current_count * (self.base_footprint * dynamic_buffer)
            
            utilization_pct = 0.0
            if area_sq_meters > 0:
                utilization_pct = (used_area / area_sq_meters) * 100
            elif est_capacity > 0:
                utilization_pct = (current_count / est_capacity) * 100
            
            # Use smoothed utilization if available
            if self.utilization_smoother:
                utilization_pct = np.mean(self.utilization_smoother)

            stats = {
                'timestamp': time.time(),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_count': current_count,
                'total_people_seen': self.total_unique_people,
                'peak_count': self.peak_occupancy,
                'capacity': est_capacity,
                'utilization_pct': round(utilization_pct, 1),
                'area_total_sqm': round(area_sq_meters, 1) if area_sq_meters else 0,
                'area_used_sqm': round(used_area, 1),
                'status': "CRITICAL" if utilization_pct > 90 else "WARNING" if utilization_pct > 70 else "NORMAL",
                'alert_active': alert_active
            }
            
            # Atomic write to avoid read errors on frontend
            temp_file = f"{self.json_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(stats, f)
            os.replace(temp_file, self.json_file)
            # print(f"DEBUG: Wrote stats to {os.path.abspath(self.json_file)}")
            
        except Exception as e:
            print(f"Warning: Could not write JSON stats: {e}")
    
    def _calculate_flow_rate(self) -> float:
        """Calculate people flow rate (new unique people per minute)"""
        if len(self.unique_history) < 30: # Need at least ~1 second of data
            return 0.0
        
        # diverse window sizes for short term flow
        window_frames = min(len(self.unique_history), self.fps * 60) # Up to 1 minute lookback
        
        current_unique = self.unique_history[-1]
        past_unique = self.unique_history[-window_frames]
        
        new_people = current_unique - past_unique
        
        # Normalize to per minute
        seconds_passed = window_frames / self.fps
        if seconds_passed <= 0:
            return 0.0
            
        flow_per_minute = (new_people / seconds_passed) * 60
        return max(0.0, flow_per_minute)
    
    def _trigger_alert(self, count: int):
        """Trigger overcrowding alert"""
        alert_msg = f"⚠️ OVERCROWDING ALERT: {count} people (threshold: {self.alert_threshold})"
        
        # Console alert
        if self.log_to_console:
            print("\n" + "=" * 60)
            print(alert_msg)
            print("=" * 60 + "\n")
        
        # Sound alert (optional)
        if self.alert_sound:
            try:
                # Simple beep (works on most systems)
                print('\a')  # ASCII bell character
            except:
                pass
        
        # Webhook alert (optional)
        if self.alert_webhook:
            self._send_webhook_alert(count)
    
    def _send_webhook_alert(self, count: int):
        """Send alert to webhook URL"""
        try:
            import requests
            payload = {
                'alert_type': 'overcrowding',
                'count': count,
                'threshold': self.alert_threshold,
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat()
            }
            requests.post(self.alert_webhook, json=payload, timeout=5)
            print(f"✓ Alert sent to webhook")
        except Exception as e:
            print(f"Warning: Could not send webhook alert: {e}")
    
    def get_summary_statistics(self) -> dict:
        """Get summary statistics for current session"""
        uptime = time.time() - self.start_time
        
        stats = {
            'session': {
                'uptime_seconds': uptime,
                'uptime_formatted': self._format_duration(uptime),
                'start_time': datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')
            },
            'occupancy': {
                'current': self.occupancy_history[-1] if self.occupancy_history else 0,
                'peak': self.peak_occupancy,
                'average': np.mean(self.occupancy_history) if self.occupancy_history else 0,
                'unique_total': self.total_unique_people
            },
            'performance': {
                'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
                'current_fps': self.fps_history[-1] if self.fps_history else 0
            },
            'alerts': {
                'total_alerts': self.alert_count,
                'alert_threshold': self.alert_threshold
            }
        }
        
        return stats
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def print_summary(self):
        """Print summary statistics to console"""
        stats = self.get_summary_statistics()
        
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(f"Started:            {stats['session']['start_time']}")
        print(f"Uptime:             {stats['session']['uptime_formatted']}")
        print(f"\nCurrent Occupancy:  {stats['occupancy']['current']}")
        print(f"Peak Occupancy:     {stats['occupancy']['peak']}")
        print(f"Average Occupancy:  {stats['occupancy']['average']:.1f}")
        print(f"Unique People:      {stats['occupancy']['unique_total']}")
        print(f"\nProcessing FPS:     {stats['performance']['avg_fps']:.1f}")
        print(f"Total Alerts:       {stats['alerts']['total_alerts']}")
        print("=" * 70 + "\n")
    
    def draw_hud(self, frame: np.ndarray, current_count: int, unique_count: int):
        """
        Draw heads-up display with key metrics on frame
        
        Args:
            frame: Frame to draw on
            current_count: Current people count
            unique_count: Total unique people count
        """
        import cv2
        
        # HUD background
        hud_height = 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (450, hud_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Current count (large)
        cv2.putText(frame, f"Count: {current_count}", (15, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Unique count
        cv2.putText(frame, f"Unique: {unique_count}", (15, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Peak occupancy
        cv2.putText(frame, f"Peak: {self.peak_occupancy}", (15, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        if self.fps_history:
            current_fps = self.fps_history[-1]
            fps_color = (0, 255, 0) if current_fps > 15 else (0, 165, 255) if current_fps > 10 else (0, 0, 255)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (300, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Alert indicator
        if self.alerts_enabled and current_count >= self.alert_threshold:
            cv2.putText(frame, "ALERT!", (300, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # Flashing border
            if int(time.time() * 2) % 2 == 0:  # Flash at 0.5Hz
                cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                            (0, 0, 255), 5)
        
        # Timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, timestamp, (15, frame.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def close(self):
        """Close analytics engine and print final summary"""
        if self.enabled and self.log_to_console:
            self.print_summary()


class PerformanceProfiler:
    """Profile performance of different processing stages"""
    
    def __init__(self):
        self.timings = {}
        self.current_stage = None
        self.stage_start = 0
    
    def start_stage(self, stage_name: str):
        """Start timing a processing stage"""
        self.current_stage = stage_name
        self.stage_start = time.perf_counter()
    
    def end_stage(self):
        """End timing current stage"""
        if self.current_stage is None:
            return
        
        elapsed = (time.perf_counter() - self.stage_start) * 1000  # ms
        
        if self.current_stage not in self.timings:
            self.timings[self.current_stage] = []
        
        self.timings[self.current_stage].append(elapsed)
        # Keep only last 100 measurements
        self.timings[self.current_stage] = self.timings[self.current_stage][-100:]
        
        self.current_stage = None
    
    def get_report(self) -> dict:
        """Get performance report"""
        report = {}
        
        for stage, times in self.timings.items():
            report[stage] = {
                'avg_ms': np.mean(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'std_ms': np.std(times)
            }
        
        return report
    
    def print_report(self):
        """Print performance report"""
        report = self.get_report()
        
        print("\n" + "=" * 70)
        print("PERFORMANCE PROFILE")
        print("=" * 70)
        print(f"{'Stage':<25} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 70)
        
        for stage, stats in report.items():
            print(f"{stage:<25} {stats['avg_ms']:>8.2f}    {stats['min_ms']:>8.2f}    {stats['max_ms']:>8.2f}")
        
        print("=" * 70 + "\n")
