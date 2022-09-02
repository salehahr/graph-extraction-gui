from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt5.QtWidgets import QHBoxLayout, QLabel

from .Panes import Panes

if TYPE_CHECKING:
    from .DataContainer import DataContainer


class Graphics(QLabel):
    def __init__(self, data_container: DataContainer, *args, **kwargs):
        super(Graphics, self).__init__(*args, **kwargs)

        # ui elements
        self._input_pane = Panes(0, data_container)
        self._output_pane = Panes(1, data_container)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout()
        layout.addWidget(self._input_pane)
        layout.addWidget(self._output_pane)
        self.setLayout(layout)

    def display_skel_image(self) -> None:
        self._input_pane.display()

    def display_node_pos(self):
        self._output_pane.display()

    @property
    def height(self) -> int:
        return self.layout().sizeHint().height()

    @property
    def width(self) -> int:
        return self.layout().sizeHint().width()
