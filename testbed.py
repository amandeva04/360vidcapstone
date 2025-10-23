# libraries
import numpy as np
import pandas as pd

# loading dataset

class DatasetLoader:
    def __init__(self, video_path, trace_path):
        self.video_path = video_path
        self.trace_path = trace_path
    
    def load_traces(self):
        traces = pd.read_csv(self.trace_path)
        return traces
    
    def segment_video(self, tiles=(6,12), bitrates=[20,50,100,200,300]):
        print(f"Segmenting video into {tiles[0]*tiles[1]} tiles")
        return {"tiles": tiles, "bitrates": bitrates}

# mapping yaw/pitch/roll to a viewport region
class ViewportProcessor:
    def __init__(self, fov=110):
        self.fov = fov

    def compute_viewport(self, yaw, pitch, roll):
        # Convert head movement into viewport region
        return {"yaw": yaw, "pitch": pitch, "roll": roll}
    
# LSTMs and GRUs will come here

# need to add streaming, evaluation, expirement, and graphing soon