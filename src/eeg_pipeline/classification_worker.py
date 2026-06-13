# Classificaiton Worker runs background thread with the processing stages:
# Preprocessing, Feature Extraction and Classification

import numpy as np
import torch
from queue import Queue
from PyQt5.QtCore import pyqtSignal, QObject

from src.eeg_pipeline.signal_processing import RealtimeProcessor, n_channels
from src.eeg_pipeline.emotion_result import EmotionResult

class ClassificationWorker(QObject):
    
    thread_result = pyqtSignal(object)  # EmotionResult containing computation results
    all_done      = pyqtSignal()        # Signal emitted when queue drains after finish()

    def __init__(self, model, stft_n: int, sample_rate: int):
        super().__init__()
        self.model  = model
        self.stft_n = stft_n
        self.sf     = sample_rate
        self.processor = RealtimeProcessor(fs=sample_rate)
        self.queue   = Queue()
        self.running = False
        self.raw_features       = []
        self.smoothed_asymmetry = None

    def set_model(self, model):
        #Set the PyTorch model used for classification
        self.model = model

    def reset(self):
        self.processor.reset()
        self.raw_features = []
        self.smoothed_asymmetry = None

        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break

    def enqueue(self, segment: np.ndarray, timestamp: float = None):
        # Add segment + timestamp to processing queue
        self.queue.put(("segment", (segment.copy(), timestamp)))

    def finish(self):
        # Signal that there are no more segments to process
        self.queue.put(("finish", None))

    def stop(self):
        # Stop the worker, discard remaining queue items
        self.running = False
        self.queue.put(("stop", None))

    def is_empty(self) -> bool:
        return self.queue.empty()

    def run(self):
        # Start main processing loop
        self.running = True

        while self.running:
            tag, data = self.queue.get()
            if tag == "stop":
                break
            elif tag == "finish":
                self.all_done.emit()
                break
            elif tag == "segment":
                seg_data, ts = data
                self.process(seg_data, ts)

    def apply_filter(self, segment: np.ndarray) -> np.ndarray:
        # Filter segment using the RealtimeProcessor
        return self.processor.filter(segment)

    def extract_features(self, filtered: np.ndarray) -> tuple:
        # Runs an FFT to extract bandpowers, and compute DE and FAA features
        return self.processor.analyze(filtered, stft_n=self.stft_n)

    def smooth_asymmetry(self, band_powers: dict):
        # Smooth FAA over a rolling 5-second window
        raw = band_powers.get('asymmetry', 0.0)

        if self.smoothed_asymmetry is None:
            self.smoothed_asymmetry = raw
        else:
            self.smoothed_asymmetry = 0.25 * raw + 0.75 * self.smoothed_asymmetry
        band_powers['asymmetry'] = self.smoothed_asymmetry

    def classify(self, de_features: np.ndarray) -> np.ndarray:
        # Smooth DE features over a rolling 5-second window, then run the model
        self.raw_features.append(de_features)

        if len(self.raw_features) > 20:
            self.raw_features.pop(0)

        features_arr = np.array(self.raw_features)
        T = features_arr.shape[0]
        smoothed = features_arr[max(0, T - min(5, T)):T].mean(axis=0)

        if self.model is not None:
            tensor = torch.tensor(smoothed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(
                    self.model(tensor), dim=1
                ).numpy()[0]
        else:
            probs = np.full(4, 0.25)
        return probs

    def process(self, segment: np.ndarray, timestamp: float):
        # Full pipeline, provides in EmotionResult object the pipeline results
        filtered   = self.apply_filter(segment)
        de, powers = self.extract_features(filtered)
        probs      = self.classify(de)
        self.smooth_asymmetry(powers)

        self.thread_result.emit(EmotionResult(de, probs, timestamp, powers))
