from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QGridLayout, QMainWindow

from .DataContainer import DataContainer
from .Graphics import Graphics
from .Sidebar import Sidebar


class Viewer(QMainWindow):
    def __init__(self):
        super(Viewer, self).__init__()

        # defaults
        self._nodes_nn = None
        self._adj_matr_predictor = None
        self._data_container = DataContainer(self)

        # ui elements
        self._graphics = Graphics(self._data_container)
        self._sidebar = Sidebar(self._data_container)

        self._init_ui()
        self._graphics.display_skel_image()

        self.show()

    def _init_ui(self) -> None:
        self.setCentralWidget(self._graphics)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sidebar)

        layout = QGridLayout()
        self.setLayout(layout)

        self.setWindowIcon(QIcon("./assets/icon.png"))
        self.setWindowTitle("Graph Extraction")
        self.resize(self.sizeHint())

    def set_models(self, nodes_nn, adj_matr_predictor):
        self._nodes_nn = nodes_nn
        self._adj_matr_predictor = adj_matr_predictor
        self._predict()  # initial pred

    def update_skel_image(self):
        self._graphics.display_skel_image()
        self._predict()

    def update_node_pos_image(self):
        pass
        # self._graphics.display_node_pos()

    def update_predicted_graph(self):
        self._graphics.display_predicted_graph()

    def _predict(self):
        if self._nodes_nn:
            skel, pos, deg = self._nodes_nn.predict_from_skel(
                self._data_container.skel_image_tensor
            )
            self._data_container.node_pos_tensor = pos

            self._adj_matr_predictor.predict((skel, pos, deg))
            self._data_container.update_adjacency_matrix(self._adj_matr_predictor)

    def sizeHint(self) -> QSize:
        width = self._graphics.width + self._sidebar.width
        height = self._graphics.height
        return QSize(width, height)

    @property
    def current_filepath(self):
        return self._data_container.current_image_filepath
