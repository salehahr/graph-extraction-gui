from .AdjMatrPredictor import AdjMatrPredictor
from .models import EdgeNN, NodesNN

from config import num_neighbours


def load():
    nodes_nn_weights = "./models/nodes_nn"
    edge_nn_weights = "./models/edge_nn"

    nodes_nn_ = NodesNN(nodes_nn_weights)
    edge_nn_ = EdgeNN(edge_nn_weights)

    return nodes_nn_, edge_nn_


nodes_nn, edge_nn = load()
adj_matr_predictor = AdjMatrPredictor(edge_nn, num_neighbours)
