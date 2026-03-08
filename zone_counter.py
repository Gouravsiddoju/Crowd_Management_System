"""
Zone-Based Counting Module
Tracks people in defined zones and counts entry/exit flows
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class Zone:
    """Represents a counting zone"""
    name: str
    polygon: List[Tuple[int, int]]  # List of (x, y) points
    entry_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    exit_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    alert_threshold: int = 0
    color: Tuple[int, int, int] = (0, 255, 255)  # BGR color for visualization


class LineCounter:
    """Counts people crossing a line"""
    
    def __init__(self, line: Tuple[Tuple[int, int], Tuple[int, int]], name: str):
        self.line = line  # ((x1, y1), (x2, y2))
        self.name = name
        self.count = 0
        self.track_positions = {}  # {track_id: last_position_relative_to_line}
    
    def _get_side(self, point: Tuple[int, int]) -> int:
        """
        Determine which side of the line the point is on
        Returns: 1 (one side), -1 (other side), 0 (on line)
        """
        x, y = point
        (x1, y1), (x2, y2) = self.line
        
        # Cross product to determine side
        d = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        
        if d > 0:
            return 1
        elif d < 0:
            return -1
        else:
            return 0
    
    def update(self, track_id: int, position: Tuple[int, int]) -> bool:
        """
        Update track position and check if line was crossed
        
        Args:
            track_id: Track ID
            position: Current position (x, y)
            
        Returns:
            True if line was crossed, False otherwise
        """
        current_side = self._get_side(position)
        
        if current_side == 0:
            return False  # On the line, don't count
        
        if track_id in self.track_positions:
            previous_side = self.track_positions[track_id]
            
            # Crossed if signs are different
            if previous_side != current_side:
                self.count += 1
                self.track_positions[track_id] = current_side
                return True
        
        self.track_positions[track_id] = current_side
        return False
    
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
        """Draw the counting line on frame"""
        (x1, y1), (x2, y2) = self.line
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Add count label
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        label = f"{self.name}: {self.count}"
        cv2.putText(frame, label, (mid_x, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class ZoneOccupancy:
    """Tracks occupancy within a zone"""
    
    def __init__(self, zone: Zone):
        self.zone = zone
        self.polygon = np.array(zone.polygon, dtype=np.int32)
        self.current_count = 0
        self.peak_count = 0
        self.tracks_inside = set()
        
        # Dwell time tracking
        self.dwell_times = defaultdict(int)  # {track_id: frames_inside}
        self.completed_dwell_times = []  # List of completed dwell times
        
        # Alert state
        self.alert_active = False
    
    def _point_in_polygon(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside polygon"""
        x, y = point
        result = cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False)
        return result >= 0
    
    def update(self, tracks: Dict[int, Tuple[int, int]]):
        """
        Update zone occupancy with current track positions
        
        Args:
            tracks: Dict of {track_id: (center_x, center_y)}
        """
        current_inside = set()
        
        for track_id, position in tracks.items():
            if self._point_in_polygon(position):
                current_inside.add(track_id)
                self.dwell_times[track_id] += 1
        
        # Track that left the zone
        left_zone = self.tracks_inside - current_inside
        for track_id in left_zone:
            if track_id in self.dwell_times:
                self.completed_dwell_times.append(self.dwell_times[track_id])
                del self.dwell_times[track_id]
        
        self.tracks_inside = current_inside
        self.current_count = len(current_inside)
        
        if self.current_count > self.peak_count:
            self.peak_count = self.current_count
        
        # Check alert threshold
        if self.zone.alert_threshold > 0:
            self.alert_active = self.current_count >= self.zone.alert_threshold
    
    def get_average_dwell_time(self, fps: int = 30) -> float:
        """Get average dwell time in seconds"""
        if not self.completed_dwell_times:
            return 0.0
        
        avg_frames = np.mean(self.completed_dwell_times)
        return avg_frames / fps
    
    def draw(self, frame: np.ndarray, show_label: bool = True):
        """Draw zone on frame"""
        # Draw polygon
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon], self.zone.color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [self.polygon], True, self.zone.color, 2)
        
        if show_label:
            # Calculate label position (centroid of polygon)
            M = cv2.moments(self.polygon)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = self.polygon[0]
            
            # Zone info
            label = f"{self.zone.name}: {self.current_count}"
            if self.alert_active:
                label += " [ALERT]"
                color = (0, 0, 255)  # Red for alert
            else:
                color = self.zone.color
            
            # Background for better readability
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (cx - 5, cy - text_h - 5), 
                         (cx + text_w + 5, cy + 5), (0, 0, 0), -1)
            
            cv2.putText(frame, label, (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


class ZoneCounter:
    """Main zone-based counting system"""
    
    def __init__(self, zones: List[Zone], fps: int = 30):
        self.zones = zones
        self.fps = fps
        
        # Create zone occupancy trackers
        self.zone_trackers = {
            zone.name: ZoneOccupancy(zone) for zone in zones
        }
        
        # Create line counters
        self.entry_counters = {}
        self.exit_counters = {}
        
        for zone in zones:
            if zone.entry_line:
                self.entry_counters[zone.name] = LineCounter(zone.entry_line, f"{zone.name} Entry")
            if zone.exit_line:
                self.exit_counters[zone.name] = LineCounter(zone.exit_line, f"{zone.name} Exit")
        
        # Global statistics
        self.frame_count = 0
        self.last_log_time = 0
    
    def update(self, tracks: Dict[int, Tuple[int, int, Tuple[int, int, int, int]]]):
        """
        Update all zones with current tracks
        
        Args:
            tracks: Dict of {track_id: (center_x, center_y, bbox)}
        """
        self.frame_count += 1
        
        # Extract positions for occupancy
        positions = {tid: (cx, cy) for tid, (cx, cy, _) in tracks.items()}
        
        # Update zone occupancy
        for tracker in self.zone_trackers.values():
            tracker.update(positions)
        
        # Update line crossing
        for zone_name, counter in self.entry_counters.items():
            for track_id, (cx, cy, _) in tracks.items():
                if counter.update(track_id, (cx, cy)):
                    # print(f"Entry detected: {zone_name} (Track {track_id})")
                    pass
        
        for zone_name, counter in self.exit_counters.items():
            for track_id, (cx, cy, _) in tracks.items():
                if counter.update(track_id, (cx, cy)):
                    # print(f"Exit detected: {zone_name} (Track {track_id})")
                    pass
    
    def draw(self, frame: np.ndarray):
        """Draw all zones and lines on frame"""
        # Draw zones
        for tracker in self.zone_trackers.values():
            tracker.draw(frame)
        
        # Draw entry lines
        for counter in self.entry_counters.values():
            counter.draw(frame, color=(0, 255, 0))  # Green for entry
        
        # Draw exit lines
        for counter in self.exit_counters.values():
            counter.draw(frame, color=(0, 0, 255))  # Red for exit
    
    def get_statistics(self) -> dict:
        """Get comprehensive zone statistics"""
        stats = {
            'zones': {},
            'summary': {
                'total_occupancy': 0,
                'total_entries': 0,
                'total_exits': 0,
                'alerts': []
            }
        }
        
        # Zone-specific stats
        for zone_name, tracker in self.zone_trackers.items():
            zone_stats = {
                'current_count': tracker.current_count,
                'peak_count': tracker.peak_count,
                'avg_dwell_time': tracker.get_average_dwell_time(self.fps),
                'alert_active': tracker.alert_active
            }
            
            # Add entry/exit counts
            if zone_name in self.entry_counters:
                zone_stats['entries'] = self.entry_counters[zone_name].count
                stats['summary']['total_entries'] += self.entry_counters[zone_name].count
            
            if zone_name in self.exit_counters:
                zone_stats['exits'] = self.exit_counters[zone_name].count
                stats['summary']['total_exits'] += self.exit_counters[zone_name].count
            
            stats['zones'][zone_name] = zone_stats
            stats['summary']['total_occupancy'] += tracker.current_count
            
            if tracker.alert_active:
                stats['summary']['alerts'].append(zone_name)
        
        return stats
    
    def draw_statistics_panel(self, frame: np.ndarray):
        """Draw statistics panel on frame"""
        panel_x = 10
        panel_y = frame.shape[0] - 200
        
        # Background
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + 350, frame.shape[0] - 10), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + 350, frame.shape[0] - 10), 
                     (255, 255, 255), 2)
        
        # Title
        y_offset = panel_y + 25
        cv2.putText(frame, "Zone Statistics", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30
        
        # Zone info
        for zone_name, tracker in self.zone_trackers.items():
            text = f"{zone_name}: {tracker.current_count}"
            color = (0, 0, 255) if tracker.alert_active else (255, 255, 255)
            cv2.putText(frame, text, (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
            
            # Entry/Exit counts
            if zone_name in self.entry_counters:
                entry_text = f"  In: {self.entry_counters[zone_name].count}"
                cv2.putText(frame, entry_text, (panel_x + 20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y_offset += 18
            
            if zone_name in self.exit_counters:
                exit_text = f"  Out: {self.exit_counters[zone_name].count}"
                cv2.putText(frame, exit_text, (panel_x + 20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                y_offset += 18


def create_zone_from_points(name: str, points: List[Tuple[int, int]], **kwargs) -> Zone:
    """Helper to create a zone from points"""
    return Zone(name=name, polygon=points, **kwargs)


def create_rectangle_zone(name: str, x: int, y: int, width: int, height: int, **kwargs) -> Zone:
    """Helper to create a rectangular zone"""
    points = [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height)
    ]
    return Zone(name=name, polygon=points, **kwargs)
