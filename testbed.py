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

# streaming
class StreamingSimulator:
    def __init__(self, bandwidth=2000):
        self.bandwidth = bandwidth
    
    def fetch_tiles(self, predicted_viewport, tiles, bitrates):
        # Simulate fetching tiles for viewport
        return {"tiles_fetched": tiles, "quality": np.random.choice(bitrates)}
    
# evaluation
class QoEEvaluator:
    def __init__(self):
        pass

    def compute_metrics(self, ground_truth, prediction):
        # Example metrics (just random values for now, need to add actual ones later)
        viewport_deviation = np.random.random()
        psnr = np.random.uniform(20, 40)
        quality_var = np.random.uniform(0, 5)
        bandwidth_occ = np.random.uniform(1000, 3000)
        
        return {
            "ViewportDeviation": viewport_deviation,
            "PSNR": psnr,
            "QualityVariance": quality_var,
            "BandwidthOccupation": bandwidth_occ
        }

class Experiment:
    def __init__(self, dataset, predictor, simulator, evaluator):
        self.dataset = dataset
        self.predictor = predictor
        self.simulator = simulator
        self.evaluator = evaluator
    
    def run(self):
        traces = self.dataset.load_traces()
        for i, row in traces.iterrows():
            gt_viewport = ViewportProcessor().compute_viewport(row['yaw'], row['pitch'], row['roll'])
            pred_viewport = self.predictor.predict(traces.iloc[:i+1])
            sim_result = self.simulator.fetch_tiles(pred_viewport, (6,12), [20,50,100,200,300])
            metrics = self.evaluator.compute_metrics(gt_viewport, pred_viewport)
            print(f"Step {i}: {metrics}")

# main
if __name__ == "__main__":
    dataset = DatasetLoader("video.mp4", "traces.csv")
    # predictor = ViewportPredictor() # model when it's ready
    simulator = StreamingSimulator()
    evaluator = QoEEvaluator()

    exp = Experiment(dataset, simulator, evaluator) # need to add model when it's ready right after dataset
    exp.run()