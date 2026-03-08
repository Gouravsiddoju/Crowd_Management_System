"""
Stream Handler for Real-Time Video Sources
Supports RTSP streams, video files, webcams, and batch processing
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import time
import threading
from queue import Queue
import warnings
warnings.filterwarnings('ignore')


class StreamHandler:
    """
    Unified interface for different video sources with automatic reconnection
    and buffering for smooth real-time processing
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.stream_type = config.get('type', 'file')
        self.cap = None
        self.connected = False
        self.frame_count = 0
        self.fps = 30
        self.frame_width = 1920
        self.frame_height = 1080
        
        # For RTSP reconnection
        self.rtsp_url = config.get('rtsp_url', '')
        self.reconnect_attempts = config.get('rtsp_reconnect_attempts', 5)
        self.reconnect_delay = config.get('rtsp_reconnect_delay', 5)
        self.transport = config.get('rtsp_transport', 'tcp')
        
        # For buffering
        # Only use buffering for live streams, not local files where read speed > processing speed
        self.use_buffering = config.get('rtsp_buffer_size', 0) > 0 and self.stream_type in ['rtsp', 'webcam']
        self.buffer_size = config.get('rtsp_buffer_size', 30)
        self.frame_buffer = Queue(maxsize=self.buffer_size) if self.use_buffering else None
        self.capture_thread = None
        self.stop_capture = False
        
        # Initialize stream
        self._connect()
    
    def _connect(self) -> bool:
        """Connect to video source"""
        if self.stream_type == 'rtsp':
            return self._connect_rtsp()
        elif self.stream_type == 'file':
            return self._connect_file()
        elif self.stream_type == 'webcam':
            return self._connect_webcam()
        else:
            print(f"Unknown stream type: {self.stream_type}")
            return False
    
    def _connect_file(self) -> bool:
        """Connect to video file"""
        import os
        file_path = self.config.get('file_path', '')
        
        if not file_path:
            print("Error: No file_path specified in config")
            return False
            
        # Fix relative paths from execution context
        if not os.path.isabs(file_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            abs_path = os.path.abspath(os.path.join(base_dir, file_path))
        else:
            abs_path = file_path
            
        print(f"Attempting to open video: {abs_path}")
        self.cap = cv2.VideoCapture(abs_path)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video file: {abs_path}")
            return False
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.connected = True
        print(f"✓ Connected to file: {file_path}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.fps}, Frames: {self.total_frames}")
        
        return True
    
    def _connect_webcam(self) -> bool:
        """Connect to webcam"""
        webcam_id = self.config.get('webcam_id', 0)
        
        self.cap = cv2.VideoCapture(webcam_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open webcam {webcam_id}")
            return False
        
        # Get webcam properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.connected = True
        print(f"✓ Connected to webcam {webcam_id}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.fps}")
        
        return True
    
    def _connect_rtsp(self) -> bool:
        """Connect to RTSP stream with reconnection logic"""
        if not self.rtsp_url:
            print("Error: No rtsp_url specified in config")
            return False
        
        for attempt in range(self.reconnect_attempts):
            print(f"Connecting to RTSP stream (attempt {attempt + 1}/{self.reconnect_attempts})...")
            
            # Set RTSP transport protocol
            if self.transport == 'tcp':
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            else:
                self.cap = cv2.VideoCapture(self.rtsp_url)
            
            if self.cap.isOpened():
                # Get stream properties
                self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                
                # Test read
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.connected = True
                    print(f"✓ Connected to RTSP stream: {self.rtsp_url}")
                    print(f"  Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.fps}")
                    
                    # Start buffering thread if enabled
                    if self.use_buffering:
                        self._start_capture_thread()
                    
                    return True
            
            print(f"Failed to connect, retrying in {self.reconnect_delay} seconds...")
            time.sleep(self.reconnect_delay)
        
        print("Error: Could not connect to RTSP stream after all attempts")
        return False
    
    def _start_capture_thread(self):
        """Start background thread for frame capture (reduces latency)"""
        self.stop_capture = False
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        print("✓ Started buffered capture thread")
    
    def _capture_frames(self):
        """Background thread function to continuously capture frames"""
        while not self.stop_capture:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Add to buffer (will block if buffer is full)
                    if not self.frame_buffer.full():
                        self.frame_buffer.put(frame)
                    else:
                        # Drop oldest frame and add new one
                        try:
                            self.frame_buffer.get_nowait()
                        except:
                            pass
                        self.frame_buffer.put(frame)
                else:
                    # Connection lost, try to reconnect
                    if self.stream_type == 'rtsp':
                        print("Stream connection lost, attempting reconnection...")
                        self.connected = False
                        if self._connect_rtsp():
                            print("✓ Reconnected successfully")
                        else:
                            print("Reconnection failed, stopping capture")
                            break
                    else:
                        break
            else:
                break
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from stream
        
        Returns:
            (success, frame) tuple
        """
        if not self.connected or self.cap is None:
            return False, None
        
        # If using buffering, read from buffer
        if self.use_buffering and self.frame_buffer is not None:
            try:
                frame = self.frame_buffer.get(timeout=1.0)
                self.frame_count += 1
                return True, frame
            except:
                return False, None
        
        # Direct read
        ret, frame = self.cap.read()
        
        if not ret:
            print(f"DEBUG: cap.read() returned False inside direct read. Frame Count = {self.frame_count}. stream_type = {self.stream_type}")
            # Try to reconnect for RTSP streams
            if self.stream_type == 'rtsp' and self.connected:
                print("Stream interrupted, attempting reconnection...")
                self.connected = False
                if self._connect_rtsp():
                    print("✓ Reconnected successfully")
                    return self.read()
            
            return False, None
        
        self.frame_count += 1
        return True, frame
    
    def get_properties(self) -> dict:
        """Get stream properties"""
        return {
            'fps': self.fps,
            'width': self.frame_width,
            'height': self.frame_height,
            'frame_count': self.frame_count,
            'total_frames': getattr(self, 'total_frames', 0),
            'connected': self.connected,
            'type': self.stream_type
        }
    
    def is_connected(self) -> bool:
        """Check if stream is connected"""
        return self.connected
    
    def release(self):
        """Release resources"""
        # Stop capture thread
        if self.capture_thread is not None:
            self.stop_capture = True
            self.capture_thread.join(timeout=2.0)
        
        # Release capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.connected = False
        print("Stream released")


class OutputHandler:
    """
    Handle output video/frames saving with flexible options
    """
    
    def __init__(self, config: dict, stream_props: dict):
        self.config = config
        self.save_video = config.get('save_video', False)
        self.save_frames = config.get('save_frames', False)
        
        self.video_writer = None
        self.frames_saved = 0
        
        if self.save_video:
            self._init_video_writer(stream_props)
        
        if self.save_frames:
            import os
            self.frames_path = config.get('frames_path', 'output_frames/')
            os.makedirs(self.frames_path, exist_ok=True)
            self.frames_interval = config.get('frames_interval', 30)
    
    def _init_video_writer(self, stream_props: dict):
        """Initialize video writer"""
        video_path = self.config.get('video_path', 'output_counted.mp4')
        codec = self.config.get('video_codec', 'mp4v')
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        fps = stream_props.get('fps', 30)
        width = stream_props.get('width', 1920)
        height = stream_props.get('height', 1080)
        
        self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if self.video_writer.isOpened():
            print(f"✓ Saving video to: {video_path}")
        else:
            print(f"Warning: Could not open video writer for {video_path}")
            self.save_video = False
    
    def write(self, frame: np.ndarray, frame_number: int):
        """Write frame to output"""
        # Save video
        if self.save_video and self.video_writer is not None:
            self.video_writer.write(frame)
        
        # Save frame images
        if self.save_frames and frame_number % self.frames_interval == 0:
            import os
            frame_path = os.path.join(self.frames_path, f"frame_{frame_number:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            self.frames_saved += 1
    
    def release(self):
        """Release resources"""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"✓ Video saved successfully")
        
        if self.save_frames:
            print(f"✓ Saved {self.frames_saved} frames to {self.frames_path}")


def test_stream_handler():
    """Test function for stream handler"""
    config = {
        'type': 'file',
        'file_path': 'cr.mp4'
    }
    
    handler = StreamHandler(config)
    
    if handler.is_connected():
        print("\nReading 10 frames...")
        for i in range(10):
            ret, frame = handler.read()
            if ret:
                print(f"Frame {i+1}: {frame.shape}")
            else:
                print(f"Failed to read frame {i+1}")
                break
    
    handler.release()


if __name__ == '__main__':
    test_stream_handler()
