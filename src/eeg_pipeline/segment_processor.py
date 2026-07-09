# Segment Processor — handles signal filtering and feature extraction
# Receives raw 1-second EEG segments via a queue, applies realtime filtering,
# extracts DE features + band powers, and emits the result.

import numpy as np
from queue import Queue
from PyQt5.QtCore import pyqtSignal, QObject

from src.eeg_pipeline.signal_processing import RealtimeProcessor, n_channels


class SegmentProcessor(QObject):

    # Emitted after filter + feature extraction for each segment
    segment_processed = pyqtSignal(object, object, object)  # de, band_powers, timestamp
    all_done          = pyqtSignal()

    def __init__(self, stft_n: int, sample_rate: int):
        super().__init__()
        self.stft_n    = stft_n
        self.sf        = sample_rate
        self.processor = RealtimeProcessor(fs=sample_rate)
        self.queue     = Queue()
        self.running   = False
        self.smoothed_asymmetry = None

    def reset(self):
        self.processor.reset()
        self.smoothed_asymmetry = None
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break

    def enqueue(self, segment: np.ndarray, timestamp: float = None):
        self.queue.put(("segment", (segment.copy(), timestamp)))

    def finish(self):
        self.queue.put(("finish", None))

    def stop(self):
        self.running = False
        self.queue.put(("stop", None))

    def is_empty(self) -> bool:
        return self.queue.empty()

    def run(self):
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

    def process(self, segment: np.ndarray, timestamp: float):
        filtered         = self.processor.filter(segment)
        de, band_powers  = self.processor.analyze(filtered, stft_n=self.stft_n)
        self.smooth_asymmetry(band_powers)
        self.segment_processed.emit(de, band_powers, timestamp)

    def smooth_asymmetry(self, band_powers: dict):
        # EMA smooth of FAA over a rolling 5-second window
        raw = band_powers.get('asymmetry', 0.0)
        if self.smoothed_asymmetry is None:
            self.smoothed_asymmetry = raw
        else:
            self.smoothed_asymmetry = 0.25 * raw + 0.75 * self.smoothed_asymmetry
        band_powers['asymmetry'] = self.smoothed_asymmetry
