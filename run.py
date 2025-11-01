# run.py
import sys, os
from sources.gui import MainWindow
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor

def apply_dark_palette(app):
    app.setStyle("Fusion")

    dark_palette = QPalette()

    dark_color = QColor(45, 45, 45)
    disabled_color = QColor(127, 127, 127)
    highlight_color = QColor(42, 130, 218)

    dark_palette.setColor(QPalette.ColorRole.Window, dark_color)
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, dark_color.darker())
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    #dark_palette.setColor(QPalette.ColorRole.Disabled, QPalette.ColorRole.Text, disabled_color)
    dark_palette.setColor(QPalette.ColorRole.Button, dark_color)
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    #dark_palette.setColor(QPalette.ColorRole.Disabled, QPalette.ColorRole.ButtonText, disabled_color)
    dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.ColorRole.Highlight, highlight_color)
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    app.setPalette(dark_palette)



def load_stylesheet(app, filename: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    qss_path = os.path.join(base_dir, filename)
    with open(qss_path, "r", encoding="utf-8") as f:
        app.setStyleSheet(f.read())

def main():
    app = QApplication(sys.argv)
    #app.setStyle('Fusion')
    #load_stylesheet(app, "resources/dark_theme.qss")
    apply_dark_palette(app)
    win = MainWindow()
    

    win.show()
    sys.exit(app.exec())
  
if __name__ == "__main__":
    main()

