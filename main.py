import sys
import os

if sys.platform == "win32":
    try:
        import torch
    except ImportError:
        pass
    
    torch_path = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_path):
        os.add_dll_directory(os.path.abspath(torch_path))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from src.ui.main_window import MainWindow

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    if sys.platform != "win32":
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QApplication(sys.argv)
    
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
