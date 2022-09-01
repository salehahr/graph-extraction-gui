import sys

from PyQt5.QtWidgets import QApplication

from gui import Viewer
from models import nodes_nn

if __name__ == "__main__":
    app = QApplication([])
    viewer = Viewer()
    viewer.set_models(nodes_nn)

    sys.exit(app.exec())
