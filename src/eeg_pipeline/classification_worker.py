# Classification Worker — receives pre-extracted DE features and classifies them
# Maintains a rolling history of features for temporal smoothing,
# then runs the model and emits (probs, timestamp).

import numpy as np
import torch
from queue import Queue
from PyQt5.QtCore import pyqtSignal, QObject


class ClassificationWorker(QObject):

    classification_done = pyqtSignal(object, object)  # probs (ndarray), timestamp
    all_done            = pyqtSignal()

    def __init__(self, model):
        super().__init__()
        self.model       = model
        self.queue       = Queue()
        self.running     = False
        self.raw_features = []  # rolling 20-frame DE history

    def set_model(self, model):
        self.model = model

    def reset(self):
        self.raw_features = []
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break

    def enqueue(self, de_features: np.ndarray, timestamp: float = None):
        self.queue.put(("de", (de_features, timestamp)))

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
            elif tag == "de":
                de, ts = data
                probs = self._classify(de)
                self.classification_done.emit(probs, ts)

    def _classify(self, de_features: np.ndarray) -> np.ndarray:
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
