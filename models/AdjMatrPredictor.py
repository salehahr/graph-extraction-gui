from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import tensorflow as tf

from . import neighbour_ops

from .models import get_edgenn_caller
from .utils import (
    classify,
    data_from_node_imgs,
    get_all_node_combinations,
    get_combo_inputs,
    get_combo_nodes,
    get_placeholders,
    get_update_function,
)

if TYPE_CHECKING:
    import numpy as np

    from .models import EdgeNN


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.int64)])
def nodes_not_found(degrees: tf.Tensor) -> tf.Tensor:
    return tf.where(tf.not_equal(degrees, 0))


class AdjMatrPredictor(object):
    def __init__(self, edge_nn: EdgeNN, num_neighbours: int):
        self._model: tf.keras.models.Model = edge_nn.keras_model
        self._init_num_neighbours = tf.constant(num_neighbours)
        self._num_neighbours = tf.Variable(
            initial_value=num_neighbours, trainable=False
        )

        # functions
        """Model prediction function, already traced."""
        self._predict_func: tf.types.experimental.ConcreteFunction = get_edgenn_caller(
            self._model
        )
        """Adjacency matrix update function, already traced."""
        self._update_func: tf.types.experimental.ConcreteFunction

        # current image data
        self._skel_img: tf.Tensor
        self._node_pos: tf.Tensor
        self._pos_list_xy: tf.Tensor

        """All initially available node pair combinations."""
        self._all_combos: tf.Tensor

        # placeholders
        """Adjacency matrix."""
        self._A: tf.Variable

        """Pool of combinations to choose from when generating neighbour combinations."""
        self._reduced_combos: tf.Tensor
        """Pair combinations based on number of neighbours set."""
        self._combos: tf.Tensor
        """Adjacency values corresponding to self._combos, as given by the model prediction."""
        self._adjacencies: tf.Tensor
        """Adjacency probability values corresponding to self._combos, as given by the model prediction."""
        self._adjacency_probs: tf.Tensor

        """Unique nodes fouund in self._combos."""
        self._nodes: tf.Tensor
        """Row indices where each node in self._nodes can be found in self._combos."""
        self._node_rows: tf.RaggedTensor
        """Summed adjacencies for each node in self._nodes, given the combinations self._combos."""
        self._node_adjacencies: tf.Tensor
        """Adjacency probabilities for each node in self._nodes, given the combinations self._combos."""
        self._node_adj_probs: tf.RaggedTensor
        """Degrees of each node in self._nodes."""
        self._node_degrees: tf.Tensor

        # lookup table, logger
        self._degrees_lookup: tf.Variable

        # flags
        self._stop_iterate: tf.bool

        # metrics
        self._tp = tf.keras.metrics.TruePositives()
        self._tn = tf.keras.metrics.TrueNegatives()
        self._fp = tf.keras.metrics.FalsePositives()
        self._fn = tf.keras.metrics.FalseNegatives()
        self._precision = tf.keras.metrics.Precision()
        self._recall = tf.keras.metrics.Recall()
        self._metrics = [
            self._tp,
            self._tn,
            self._fp,
            self._fn,
            self._precision,
            self._recall,
        ]

    def predict(self, input_data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> None:
        """Runs prediction only over one batch of knn pair combinations.
        Returns the time taken (neglecting the prediction made
        when graph tracing occurs)."""
        # with tracing
        self._init_prediction(*input_data)
        self._update_func(self._combos, self._adjacencies, self._A)

    def _init_prediction(
        self, skel_img: tf.Tensor, node_pos: tf.Tensor, degrees: tf.Tensor
    ) -> None:
        """Initialises placeholders and flags before predicting."""

        # store skel img, node_pos matrix
        self._skel_img = skel_img
        self._node_pos = node_pos
        self._num_neighbours.assign(self._init_num_neighbours)

        # derived data; constants/reference
        node_pos = tf.expand_dims(node_pos, -1)
        degrees = tf.expand_dims(degrees, -1)
        self._pos_list_xy, degrees_list, num_nodes = data_from_node_imgs(
            node_pos, degrees
        )
        degrees_list = tf.cast(degrees_list, tf.int64)

        self._all_combos = get_all_node_combinations(num_nodes)
        all_nodes = tf.expand_dims(tf.range(num_nodes, dtype=tf.int64), axis=-1)

        # initialise lookup values
        self._degrees_lookup, self._A = get_placeholders(degrees_list, num_nodes)
        self._update_func = get_update_function(self._A)

        # iniitial/volatile; neighbours and prediction for neighbours
        self._reduced_combos = tf.identity(self._all_combos)
        self._update_neighbours(all_nodes)

        # reset flags
        self._stop_iterate: tf.bool = tf.constant(False)

        # reset metrics
        for metric in self._metrics:
            metric.reset_state()

    def _update_neighbours(self, nodes: Optional[tf.Tensor] = None) -> None:
        """Updates node pair combos with new neighbours."""
        self._combos = neighbour_ops.get_neighbours(
            self._num_neighbours.value(),
            self._reduced_combos,
            self._pos_list_xy,
            self.nodes_not_found if nodes is None else nodes,
        )

        # predict and update nodes only if combos are found
        self._adjacency_probs, self._adjacencies = tf.cond(
            self.num_combos > tf.constant(0),
            true_fn=lambda: self._get_predictions(),
            false_fn=lambda: (None, None),
        )
        tf.cond(
            self.num_combos > tf.constant(0),
            true_fn=lambda: self._update_nodes(),
            false_fn=lambda: self._set_stop(),
        )

    def _update_nodes(self) -> None:
        """Update list of nodes after performing update on self._combos."""
        (
            self._nodes,
            self._node_rows,
            self._node_adjacencies,
            self._node_adj_probs,
            self._node_degrees,
        ) = get_combo_nodes(
            self._combos,
            self._adjacencies,
            self._adjacency_probs,
            self._degrees_lookup.value(),
        )

    # noinspection PyUnreachableCode
    def _get_predictions(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Obtains the model prediction on current neighbour node combinations.."""
        current_batch = get_combo_inputs(
            tf.squeeze(self._skel_img),
            tf.squeeze(self._node_pos),
            self._combos,
            self._pos_list_xy,
        )
        probabilities = self._predict_func(*current_batch)

        adj, adj_probs = classify(probabilities)

        # expand dimensions of (adj, adj_probs) if their lengths are one
        return tf.cond(
            tf.size(adj) > 1,
            true_fn=lambda: (adj, adj_probs),
            false_fn=lambda: (
                tf.expand_dims(adj, axis=-1),
                tf.expand_dims(adj_probs, axis=-1),
            ),
        )

    @property
    def nodes_not_found(self) -> tf.Tensor:
        return nodes_not_found(self._degrees_lookup.value())

    @property
    def num_combos(self) -> tf.Tensor:
        return tf.shape(self._combos)[0]

    def _set_stop(self) -> None:
        """Sets flag to stop iterating."""
        self._stop_iterate = tf.constant(True)

    @property
    def A(self) -> np.ndarray:
        return self._A.value().numpy()
