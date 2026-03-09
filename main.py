import sys
import os
import torch  # Import torch before PyQt5 to prevent DLL initialization conflicts
from PyQt5.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def main():
    # Suppress Qt Wayland/XCB warnings by forcing xcb
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
