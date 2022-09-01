from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QGridLayout, QMainWindow

from .config import DEFAULT_FILEPATH, IMAGE_SIZE
from .Graphics import Graphics
from .Sidebar import Sidebar


class Viewer(QMainWindow):
    def __init__(self):
        super(Viewer, self).__init__()

        # defaults
        self.image_size = IMAGE_SIZE

        # ui elements
        self._graphics = Graphics()
        self._sidebar = Sidebar(self._graphics)

        self._init_ui()
        self._graphics.display()

        self.show()

    def _init_ui(self) -> None:
        self.setCentralWidget(self._graphics)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sidebar)

        layout = QGridLayout()
        self.setLayout(layout)

        self.setWindowIcon(QIcon("./img/icon.png"))
        self.setWindowTitle("Graph Extraction")
        self.resize(self.sizeHint())

    def sizeHint(self) -> QSize:
        width = self._graphics.width + self._sidebar.width
        height = self._graphics.height
        return QSize(width, height)

    @property
    def current_filepath(self):
        if self._graphics.current_filepath:
            return self._graphics.current_filepath
        else:
            return DEFAULT_FILEPATH
