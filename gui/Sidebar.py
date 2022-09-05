from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Optional

from PyQt5.QtCore import QRunnable, Qt, QThread, QThreadPool, pyqtSlot
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from models.ACS import AdjCombinationSchemes

if TYPE_CHECKING:
    from .DataContainer import DataContainer

from config import IMAGE_DIRECTORY


class Sidebar(QDockWidget):
    def __init__(self, data_container: DataContainer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._file_browser = FileBrowser("File browser", data_container)
        self._adj_scheme = AdjacencyScheme("AC scheme settings", data_container)

        self._init_layout()

        self.width: int = self.sizeHint().width()
        self.height: int = self.sizeHint().height()

    def _init_layout(self) -> None:
        # remove title bar
        self.setTitleBarWidget(QWidget())
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)

        layout = QVBoxLayout()
        layout.addWidget(self._file_browser)
        layout.addWidget(self._adj_scheme)

        multi_widget = QWidget()
        multi_widget.setLayout(layout)

        self.setWidget(multi_widget)

        layout.setAlignment(Qt.AlignTop)

    def reset_settings(self):
        self._adj_scheme.reset_settings()


class SideBarWidget(QGroupBox):
    def __init__(self, title: str, data_container: DataContainer, *args, **kwargs):
        super().__init__(title, *args, **kwargs)

        self._data_container: DataContainer = data_container

        self._init_layout()

    def _init_layout(self):
        layout = QGridLayout()
        self.setLayout(layout)


class FileBrowser(SideBarWidget):
    def __init__(self, *args, **kwargs):
        self._directory_field: Optional[QLineEdit] = None
        self._current_directory: str = IMAGE_DIRECTORY

        super().__init__(*args, **kwargs)

    def _init_layout(self):
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse)

        self._directory_field = QLineEdit(self.current_directory)
        self._directory_field.returnPressed.connect(self._update_directory)

        layout = QGridLayout()
        layout.addWidget(QLabel("Directory:"), 0, 0)
        layout.addWidget(self._directory_field, 1, 0)
        layout.addWidget(browse_button, 2, 0)

        self.setLayout(layout)

    def _browse(self):
        self._update_directory()

        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setDirectory(os.path.abspath(self.current_directory))
        dialog.setNameFilter("Images (*.png)")
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_():
            filenames = dialog.selectedFiles()
            self.current_filepath = filenames[0]

    def _update_directory(self):
        new_dir = self._directory_field.text()

        if os.path.exists(new_dir):
            self.current_directory = new_dir

    @property
    def current_directory(self):
        return self._current_directory

    @current_directory.setter
    def current_directory(self, new_dir):
        if os.path.abspath(self._current_directory) == os.path.abspath(new_dir):
            return

        self._current_directory = new_dir
        self._directory_field.setText(self.current_directory)

    @property
    def current_filepath(self):
        return self._data_container.current_image_filepath

    @current_filepath.setter
    def current_filepath(self, filepath):
        self._data_container.current_image_filepath = filepath
        self.current_directory = os.path.split(
            os.path.relpath(self.current_filepath, start="./")
        )[0]


class ACSWorker(QRunnable):
    def __init__(self, function: Callable, *args, **kwargs):
        super(ACSWorker, self).__init__(*args, **kwargs)
        self._function = function

    @pyqtSlot()
    def run(self):
        self._function()


class AdjacencyScheme(SideBarWidget):
    def __init__(self, *args, **kwargs):
        self._num_neighbours_field: Optional[QLineEdit] = None
        self._algorithm_field: Optional[QComboBox] = None
        self._button: QPushButton = QPushButton("Apply")

        super().__init__(*args, **kwargs)

        self._thread_pool = QThreadPool()
        self._worker = ACSWorker(self._data_container.update_adjacency_matrix)
        self.predictor.finished.connect(self._enable_button)

    def _init_layout(self):
        validator = QIntValidator(1, 50)
        self._num_neighbours_field = QLineEdit()
        self._num_neighbours_field.setValidator(validator)
        self._num_neighbours_field.setFixedWidth(50)

        self._algorithm_field = QComboBox()
        for algorithm in AdjCombinationSchemes:
            self._algorithm_field.addItem(algorithm.name)
        self._num_neighbours_field.setFixedWidth(50)

        self.reset_settings()

        layout = QFormLayout()
        layout.addRow("Algorithm", self._algorithm_field)
        layout.addRow("Num. neighbours (k0)", self._num_neighbours_field)

        self._button.clicked.connect(self._apply_settings)
        layout.addRow(self._button)

        self.setLayout(layout)

    def _disable_button(self) -> None:
        self._button.setDisabled(True)

    def _enable_button(self) -> None:
        self._button.setEnabled(True)

    def reset_settings(self):
        self._algorithm_field.setCurrentIndex(self.algorithm)
        self._num_neighbours_field.setText(str(self.num_neighbours))

    def _apply_settings(self):
        new_algorithm = int(self._algorithm_field.currentIndex())
        new_k0 = int(self._num_neighbours_field.text())

        algorithm_changed = new_algorithm != self.algorithm
        k0_changed = new_k0 != self.num_neighbours

        if not (k0_changed or algorithm_changed):
            return
        self._disable_button()

        if algorithm_changed:
            self.algorithm = new_algorithm
        if k0_changed:
            self.num_neighbours = new_k0

        self._thread_pool.start(self._worker)

    @property
    def predictor(self):
        return self._data_container.predictor

    @property
    def algorithm(self) -> int:
        return self._data_container.algorithm

    @algorithm.setter
    def algorithm(self, value: int):
        self._data_container.algorithm = value

    @property
    def num_neighbours(self) -> int:
        return self._data_container.num_neighbours

    @num_neighbours.setter
    def num_neighbours(self, k: int):
        self._data_container.num_neighbours = k
