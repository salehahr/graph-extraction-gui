from .models import NodesNN


def load():
    img_dims = (256, 256)

    nodes_nn_weights = "./models/nodes_nn"
    nodes_nn_ = NodesNN(nodes_nn_weights)

    return nodes_nn_


nodes_nn = load()
