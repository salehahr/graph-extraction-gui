from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import tensorflow as tf
from PyQt5.QtGui import QPixmap

from .graphics_ops import tf_img_to_pixmap, fp_to_grayscale_img

if TYPE_CHECKING:
    import numpy as np
    from .Viewer import Viewer

from .config import DEFAULT_FILEPATH, IMAGE_SIZE


class DataContainer(object):
    def __init__(self, viewer: Viewer):
        self.image_size = IMAGE_SIZE
        self._viewer = viewer

        self._current_image_filepath: str = DEFAULT_FILEPATH
        self._node_pos_tensor: Optional[tf.Tensor] = None

        self._pos_list_xy: Optional[np.ndarray] = None
        self._adjacency_matrix: Optional[np.ndarray] = None

    def update_adjacency_matrix(self, predictor):
        self._pos_list_xy = predictor.pos_list_xy
        self._adjacency_matrix = predictor.A
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
    def pos_list_xy(self):
        return self._pos_list_xy

    @property
    def adjacency_matrix(self) -> np.ndarray:
        return self._adjacency_matrix
