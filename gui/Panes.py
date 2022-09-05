from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QLabel, QVBoxLayout

from .graphics_ops import draw_graph

if TYPE_CHECKING:
    from .DataContainer import DataContainer


class PanesEnum(Enum):
    INPUT = 0
    OUTPUT = 1


class Panes(QLabel):
    def __init__(self, _id: int, data_container: DataContainer, *args, **kwargs):
        super(Panes, self).__init__(*args, **kwargs)
        self.type = PanesEnum(_id)
        self.id = _id

        self._data_container = data_container
        self._image_height: int = data_container.image_size
        self._text_height: int = 20
        self._descriptor = "Input" if self.type == PanesEnum.INPUT else "Output"

        # ui elements
        self._image: Optional[QLabel] = None
        self._text: Optional[QLabel] = None

        self._init_layout()
        self.setFixedWidth(self.width)
        self.setFixedHeight(self.height)

    def _init_layout(self) -> None:
        self._image = QLabel()
        self._image.setFixedWidth(self._image_height)
        self._image.setFixedHeight(self._image_height)

        self._text = QLabel()
        self._text.setFixedHeight(self._text_height)

        descriptor = QLabel(self._descriptor)
        descriptor.setFixedHeight(self._text_height)

        layout = QVBoxLayout()
        layout.addWidget(descriptor)
        layout.addWidget(self._image)
        layout.addWidget(self._text)
        self.setLayout(layout)

    def display(self) -> None:
        self._display_image()
        self._display_filename()

    def _display_image(self) -> None:
        if self.type == PanesEnum.INPUT:
            self._image.setPixmap(self._data_container.skel_image)
        else:
            self._image.setPixmap(self._data_container.node_pos_image)

    def _display_filename(self):
        if self.type == PanesEnum.OUTPUT:
            return

        text = os.path.split(self._data_container.current_image_filepath)[1]
        text = os.path.splitext(text)[0]

        self._text.setText(text)

    def display_predicted_graph(self):
        if self.type == PanesEnum.INPUT:
            return

        A = self._data_container.adjacency_matrix
        A = A.squeeze() if A.ndim == 3 else A

        skel_image = self._data_container.skel_image_array
        pos_list = self._data_container.pos_list_xy

        img = draw_graph(skel_image, pos_list, A)
        self._image.setPixmap(img)

    def sizeHint(self) -> QSize:
        width = self._image_height
        height = self._image_height + self._text_height
        return QSize(width, height)

    @property
    def width(self) -> int:
        return self.layout().sizeHint().width()

    @property
    def height(self) -> int:
        return self.layout().sizeHint().height()
