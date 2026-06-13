from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox, QSpinBox
from PyQt5.QtCore import pyqtSignal


class ChannelSelectorWidget(QWidget):
    """Reusable channel selection widget with Single / Range / All modes."""
    selection_changed = pyqtSignal()
    
    def __init__(self, max_channels=256, parent=None):
        super().__init__(parent)
        self._max_channels = max_channels
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Channels:"))
        
        self.channel_mode_combo = QComboBox()
        self.channel_mode_combo.addItems(["Single Channel", "Channel Range", "All Channels"])
        self.channel_mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.channel_mode_combo)
        
        self.label_ch_from = QLabel("Ch:")
        layout.addWidget(self.label_ch_from)
        self.spin_ch_from = QSpinBox()
        self.spin_ch_from.setRange(1, self._max_channels)
        self.spin_ch_from.setValue(1)
        self.spin_ch_from.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.spin_ch_from)
        
        self.label_ch_to = QLabel("to:")
        layout.addWidget(self.label_ch_to)
        self.spin_ch_to = QSpinBox()
        self.spin_ch_to.setRange(1, self._max_channels)
        self.spin_ch_to.setValue(5)
        self.spin_ch_to.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.spin_ch_to)
        
        self.label_ch_to.setVisible(False)
        self.spin_ch_to.setVisible(False)
        layout.addStretch()
    
    def set_max_channels(self, max_channels):
        self._max_channels = max_channels
        self.spin_ch_from.setMaximum(max_channels)
        self.spin_ch_to.setMaximum(max_channels)
    
    def get_selected_channels(self, max_channels=None):
        limit = max_channels if max_channels is not None else self._max_channels
        mode = self.channel_mode_combo.currentIndex()
        if mode == 0:
            ch = self.spin_ch_from.value() - 1
            return [(f"Ch {ch+1}", ch)] if ch < limit else []
        elif mode == 1:
            ch_from = self.spin_ch_from.value() - 1
            ch_to = self.spin_ch_to.value() - 1
            if ch_from > ch_to:
                ch_from, ch_to = ch_to, ch_from
            return [(f"Ch {i+1}", i) for i in range(ch_from, min(ch_to + 1, limit))]
        else:
            return [(f"Ch {i+1}", i) for i in range(limit)]
    
    def _on_mode_changed(self, index):
        is_range = (index == 1)
        is_all = (index == 2)
        self.label_ch_from.setVisible(not is_all)
        self.spin_ch_from.setVisible(not is_all)
        self.label_ch_to.setVisible(is_range)
        self.spin_ch_to.setVisible(is_range)
        self.label_ch_from.setText("Ch:" if not is_range else "From:")
        self.selection_changed.emit()
    
    def _on_value_changed(self):
        self.selection_changed.emit()
