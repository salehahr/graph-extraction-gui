from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QGridLayout, QMainWindow

from .DataContainer import DataContainer
from .Graphics import Graphics
from .Sidebar import Sidebar


class Viewer(QMainWindow):
    def __init__(self):
        super(Viewer, self).__init__()

        # data elements
        self._data_container = DataContainer(self)

        # ui elements
        self._graphics = Graphics(self._data_container)
        self._sidebar = Sidebar(self._data_container)

        self._init_ui()
        self.show()

        # initial prediction
        self.update_skel_image()

    def _init_ui(self) -> None:
        self.setCentralWidget(self._graphics)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sidebar)

        layout = QGridLayout()
        self.setLayout(layout)

        self.setWindowIcon(QIcon("./assets/icon.png"))
        self.setWindowTitle("Graph Extraction")
        self.resize(self.sizeHint())

    def update_skel_image(self):
        self._graphics.display_skel_image()
        self._data_container.new_image_prediction()

    def update_node_pos_image(self):
        pass
        # self._graphics.display_node_pos()

    def update_predicted_graph(self):
        self._graphics.display_predicted_graph()

    def reset_settings(self):
        self._sidebar.reset_settings()

    def sizeHint(self) -> QSize:
        width = self._graphics.width + self._sidebar.width
        height = self._graphics.height
        return QSize(width, height)
