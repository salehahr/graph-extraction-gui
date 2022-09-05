from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from .AdjMatrPredictor import AdjMatrPredictor
from .models import EdgeNN, NodesNN

if TYPE_CHECKING:
    from gui.DataContainer import DataContainer


def load_models(data_container: DataContainer) -> Tuple[NodesNN, AdjMatrPredictor]:
    nodes_nn_weights = "./models/nodes_nn"
    edge_nn_weights = "./models/edge_nn"

    nodes_nn = NodesNN(nodes_nn_weights)
    edge_nn = EdgeNN(edge_nn_weights)

    adj_matr_predictor = AdjMatrPredictor(edge_nn, data_container)

    return nodes_nn, adj_matr_predictor
