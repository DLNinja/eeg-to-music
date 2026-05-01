import sys
import os

# Fix for Windows DLL loading issue (PyTorch + PyQt5)
if sys.platform == "win32":
    # Importing torch before PyQt5 is CRITICAL on Windows to avoid DLL conflicts (c10.dll)
    try:
        import torch
    except ImportError:
        pass
    
    # Add torch lib to DLL search path
    torch_path = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_path):
        os.add_dll_directory(os.path.abspath(torch_path))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from src.ui.main_window import MainWindow

def main():
    # Enable high DPI scaling for Windows/4K displays
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Suppress Qt Wayland/XCB warnings by forcing xcb (Linux only)
    if sys.platform != "win32":
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QApplication(sys.argv)
    
    # Set application style (optional, makes it look more native across platforms)
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
