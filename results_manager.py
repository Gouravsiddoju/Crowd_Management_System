import os
from datetime import datetime
import shutil

class ResultManager:
    """
    Manages the creation of result directories and file organization.
    Creates a structure like: results/2023-10-27_10-30-00/...
    """
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(self.base_dir, self.timestamp)
        self.ensure_dir(self.run_dir)
        print(f"✓ Results directory created: {self.run_dir}")

    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_path(self, filename):
        """Get full path for a file inside the run directory"""
        return os.path.join(self.run_dir, filename)

    def save_config(self, config_path):
        """Copy config file to results for reproducibility"""
        if os.path.exists(config_path):
            shutil.copy(config_path, self.run_dir)
            
    def get_run_dir(self):
        return self.run_dir
