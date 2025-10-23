import numpy as np
import pandas as pd

# -------------------------------
# 1. DATASET LOADER
# -------------------------------
class DatasetLoader:
    def __init__(self, video_path, trace_path):
        self.video_path = video_path
        self.trace_path = trace_path
    
    def load_traces(self):
        # Load head movement traces: yaw, pitch, roll
        traces = pd.read_csv(self.trace_path)
        return traces
    
    def segment_video(self, tiles=(6,12), bitrates=[20,50,100,200,300]):
        # Create video segments by tiles and encoding levels
        print(f"Segmenting video into {tiles[0]*tiles[1]} tiles")
        return {"tiles": tiles, "bitrates": bitrates}

# -------------------------------
# 2. VIEWPORT GROUND TRUTH
# -------------------------------
class ViewportProcessor:
    def __init__(self, fov=110):
        self.fov = fov  # field of view in degrees

    def compute_viewport(self, yaw, pitch, roll):
        # Convert head movement into viewport region
        return {"yaw": yaw, "pitch": pitch, "roll": roll}