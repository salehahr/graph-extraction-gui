from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import tensorflow as tf
from PyQt5.QtGui import QPixmap

from models import load_models

from .graphics_ops import fp_to_grayscale_img, tf_img_to_pixmap

if TYPE_CHECKING:
    import numpy as np
    from .Viewer import Viewer
    from models.AdjMatrPredictor import AdjMatrPredictor
    from models.models import NodesNN

from config import DEFAULT_FILEPATH, IMAGE_SIZE, algorithm, num_neighbours


class DataContainer(object):
    def __init__(self, viewer: Viewer):
        self.image_size: int = IMAGE_SIZE

        # predictor parameters
        self._algorithm: int = algorithm.value
        self._num_neighbours: int = num_neighbours

        # data objects
        nodes_nn, predictor = load_models(self)
        self._viewer: Viewer = viewer
        self._nodes_nn: NodesNN = nodes_nn
        self._predictor: AdjMatrPredictor = predictor

        # predictor inputs
        self._current_image_filepath: str = DEFAULT_FILEPATH
        self._node_pos_tensor: Optional[tf.Tensor] = None
        self._node_deg_tensor: Optional[tf.Tensor] = None

    def new_image_prediction(self):
        if self._nodes_nn:
            (
                _,
                self.node_pos_tensor,
                self.node_deg_tensor,
            ) = self._nodes_nn.predict_from_skel(self.skel_image_tensor)

            self._update_adjacency_matrix()

    def _update_adjacency_matrix(self):
        self._predictor.predict()
        self._viewer.update_predicted_graph()

    @property
    def current_image_filepath(self) -> str:
        return self._current_image_filepath

    @current_image_filepath.setter
    def current_image_filepath(self, filepath: str) -> None:
        self._current_image_filepath = filepath
        self._viewer.update_skel_image()

    # predictor inputs
    @property
    def skel_image(self) -> QPixmap:
        return QPixmap(self.current_image_filepath)

    @property
    def skel_image_tensor(self) -> tf.Tensor:
        return fp_to_grayscale_img(self.current_image_filepath)

    @property
    def skel_image_array(self) -> np.ndarray:
        return self.skel_image_tensor.numpy()

    @property
    def node_pos_tensor(self) -> tf.Tensor:
        return self._node_pos_tensor

    @node_pos_tensor.setter
    def node_pos_tensor(self, value: tf.Tensor) -> None:
        self._node_pos_tensor = value
        self._viewer.update_node_pos_image()

    @property
    def node_pos_image(self) -> QPixmap:
        return tf_img_to_pixmap(self.node_pos_tensor)

    @property
    def node_deg_tensor(self) -> tf.Tensor:
        return self._node_deg_tensor

    @node_deg_tensor.setter
    def node_deg_tensor(self, value: tf.Tensor) -> None:
        self._node_deg_tensor = value

    # predictor parameters
    @property
    def algorithm(self) -> int:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: int) -> None:
        self._algorithm = value

    @property
    def num_neighbours(self) -> int:
        return self._num_neighbours

    @num_neighbours.setter
    def num_neighbours(self, k: int) -> None:
        if self._num_neighbours != k:
            self._num_neighbours = k
            self._update_adjacency_matrix()

    # predictor outputs
    @property
    def pos_list_xy(self):
        return self._predictor.pos_list_xy

    @property
    def adjacency_matrix(self) -> np.ndarray:
        return self._predictor.A
