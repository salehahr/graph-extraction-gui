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
        self._nodes_nn = None
        self._adj_matr_predictor = None

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

    def set_models(self, nodes_nn, adj_matr_predictor):
        self._nodes_nn = nodes_nn
        self._adj_matr_predictor = adj_matr_predictor
        self.predict()  # initial pred

    def predict(self):
        if self._nodes_nn:
            skel, pos, deg = self._nodes_nn.predict_from_fp(self.current_filepath)
            self._adj_matr_predictor.predict((skel, pos, deg))

            self._graphics.display_node_pos(pos)

    @property
    def current_filepath(self):
        if self._graphics.current_filepath:
            return self._graphics.current_filepath
        else:
            return DEFAULT_FILEPATH
