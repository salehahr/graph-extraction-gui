from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import tensorflow as tf
from PyQt5.QtGui import QPixmap

from models import adj_matr_predictor, nodes_nn

from .graphics_ops import fp_to_grayscale_img, tf_img_to_pixmap

if TYPE_CHECKING:
    import numpy as np
    from .Viewer import Viewer
    from models.AdjMatrPredictor import AdjMatrPredictor
    from models.models import NodesNN

from config import DEFAULT_FILEPATH, IMAGE_SIZE, num_neighbours


class DataContainer(object):
    def __init__(self, viewer: Viewer):
        self.image_size: int = IMAGE_SIZE

        # data objects
        self._viewer: Viewer = viewer
        self._nodes_nn: NodesNN = nodes_nn
        self._predictor: AdjMatrPredictor = adj_matr_predictor

        # predictor parameters
        self._num_neighbours: int = num_neighbours

        # predictor inputs
        self._current_image_filepath: str = DEFAULT_FILEPATH
        self._node_pos_tensor: Optional[tf.Tensor] = None
        self._node_deg_tensor: Optional[tf.Tensor] = None

        # predictor outputs
        self._pos_list_xy: Optional[np.ndarray] = None
        self._adjacency_matrix: Optional[np.ndarray] = None

    def new_image_prediction(self):
        if self._nodes_nn:
            self.predictor_inputs = self._nodes_nn.predict_from_skel(
                self.skel_image_tensor
            )
            self._update_adjacency_matrix()

    def _update_adjacency_matrix(self):
        self._predictor.predict(self.predictor_inputs)

        self._pos_list_xy = self._predictor.pos_list_xy
        self._adjacency_matrix = self._predictor.A

        self._viewer.update_predicted_graph()

    @property
    def current_image_filepath(self) -> str:
        return self._current_image_filepath

    @current_image_filepath.setter
    def current_image_filepath(self, filepath: str) -> None:
        self._current_image_filepath = filepath
        self._viewer.update_skel_image()

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

    @property
    def predictor_inputs(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.skel_image_tensor, self.node_pos_tensor, self.node_deg_tensor

    @predictor_inputs.setter
    def predictor_inputs(self, inputs: Tuple[tf.Tensor, ...]):
        _, node_pos, node_deg = inputs

        self.node_pos_tensor = node_pos
        self.node_deg_tensor = node_deg

    @property
    def pos_list_xy(self):
        return self._pos_list_xy

    @property
    def adjacency_matrix(self) -> np.ndarray:
        return self._adjacency_matrix

    @property
    def num_neighbours(self):
        return self._num_neighbours

    @num_neighbours.setter
    def num_neighbours(self, k: int):
        if self._num_neighbours != k:
            self._num_neighbours = k

            self._predictor.k0 = k
            self._update_adjacency_matrix()
