import os
import torch
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox
from PyQt5.QtCore import pyqtSignal
from src.eeg_pipeline.emotion_classifier import MODEL_ARCHITECTURES, load_emotion_model, DEFAULT_MODEL_PATH

class ModelSelectorWidget(QWidget):
    """Horizontal selector for choosing model architecture and checkpoint files."""
    model_loaded = pyqtSignal(object, str, str)  # model_instance, arch_name, checkpoint_path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.refresh_checkpoints()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(QLabel("Architecture:"))
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(list(MODEL_ARCHITECTURES.keys()))
        self.arch_combo.setMinimumWidth(120)
        self.arch_combo.currentTextChanged.connect(self.refresh_checkpoints)
        layout.addWidget(self.arch_combo)
        
        layout.addWidget(QLabel("Checkpoint:"))
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.setMinimumWidth(180)
        layout.addWidget(self.checkpoint_combo)
        
        self.load_btn = QPushButton("Load Model")
        self.load_btn.setObjectName("loadModelBtn")
        self.load_btn.clicked.connect(self.load_selected_model)
        layout.addWidget(self.load_btn)
        
    def refresh_checkpoints(self):
        self.checkpoint_combo.clear()
        arch = self.arch_combo.currentText()
        if not arch:
            return
        classifiers_dir = os.path.join("models/classifiers", arch)
        if not os.path.exists(classifiers_dir):
            os.makedirs(classifiers_dir)
            
        pt_files = sorted(f for f in os.listdir(classifiers_dir) if f.endswith(".pt"))

        # Always pre-select the preferred checkpoint (matching DEFAULT_MODEL_PATH)
        preferred = os.path.basename(DEFAULT_MODEL_PATH)
        if preferred in pt_files:
            pt_files.remove(preferred)
            pt_files.insert(0, preferred)

        self.checkpoint_combo.addItems(pt_files)
        
    def load_selected_model(self):
        arch = self.arch_combo.currentText()
        chk = self.checkpoint_combo.currentText()
        if not chk:
            QMessageBox.warning(self, "No Checkpoint", f"No model checkpoints (.pt) found in models/classifiers/{arch}/.")
            return
            
        chk_path = os.path.join("models/classifiers", arch, chk)
        model = load_emotion_model(model_path=chk_path, arch_name=arch)
        if model is not None:
            self.model_loaded.emit(model, arch, chk_path)
            QMessageBox.information(self, "Model Loaded", f"Successfully loaded {arch} with checkpoint {chk}.")
        else:
            QMessageBox.critical(self, "Load Error", f"Failed to load {arch} from {chk_path}.")
