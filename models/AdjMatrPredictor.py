from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal

from . import neighbour_ops, utils
from .ACS import AdjCombinationSchemes
from .models import get_edgenn_caller

if TYPE_CHECKING:
    import numpy as np

    from gui.DataContainer import DataContainer

    from .models import EdgeNN


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.int64)])
def nodes_not_found(degrees: tf.Tensor) -> tf.Tensor:
    return tf.where(tf.not_equal(degrees, 0))


class AdjMatrPredictor(QObject):
    finished = pyqtSignal()

    def __init__(self, edge_nn: EdgeNN, data_container: DataContainer, *args, **kwargs):
        super(AdjMatrPredictor, self).__init__(*args, **kwargs)

        self._model: tf.keras.models.Model = edge_nn.keras_model
        self._data_container: DataContainer = data_container

        self._num_neighbours = tf.Variable(
            initial_value=self._data_container.num_neighbours, trainable=False
        )

        # functions
        """Model prediction function, already traced."""
        self._predict_func: tf.types.experimental.ConcreteFunction = get_edgenn_caller(
            self._model
        )
        """Adjacency matrix update function, already traced."""
        self._update_func: tf.types.experimental.ConcreteFunction

        # current image data
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

    def predict(self) -> None:
        """Runs prediction only over one batch of knn pair combinations.
        Returns the time taken (neglecting the prediction made
        when graph tracing occurs)."""
        # with tracing
        self._init_prediction()

        if self.algorithm == AdjCombinationSchemes.BAC:
            self._update_func(self._combos, self._adjacencies, self._A)

        elif self.algorithm == AdjCombinationSchemes.LAC:
            num_iters = tf.Variable(initial_value=0, trainable=False)
            while not self._stop_iterate:
                self._predict_ok_good()
                self._predict_bad()

                self._increase_neighbours()
                num_iters.assign_add(1)

        self.finished.emit()

    def _init_prediction(self) -> None:
        """Initialises placeholders and flags before predicting."""
        self._num_neighbours.assign(self.k0)

        # derived data; constants/reference
        node_pos = tf.expand_dims(self.node_pos, -1)
        degrees = tf.expand_dims(self.node_degrees, -1)
        self._pos_list_xy, degrees_list, num_nodes = utils.data_from_node_imgs(
            node_pos, degrees
        )
        degrees_list = tf.cast(degrees_list, tf.int64)

        self._all_combos = utils.get_all_node_combinations(num_nodes)
        all_nodes = tf.expand_dims(tf.range(num_nodes, dtype=tf.int64), axis=-1)

        # initialise lookup values
        self._degrees_lookup, self._A = utils.get_placeholders(degrees_list, num_nodes)
        self._update_func = utils.get_update_function(self._A)

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

    def _increase_neighbours(self) -> None:
        """Doubles the number of neighbours to be searched and
        geerates new neighbour pair combinations.
        Sets stop flag if the newly generated combinations have
        the same length as the previous combinations.
        """
        num_old_combos = tf.shape(self._combos)[0]

        # noinspection PyTypeChecker
        self._num_neighbours.assign(self._num_neighbours * 2)
        self._update_neighbours()

        tf.cond(
            (num_old_combos == self.num_combos),
            true_fn=lambda: self._set_stop(),
            false_fn=lambda: None,
        )

    def _update_nodes(self) -> None:
        """Update list of nodes after performing update on self._combos."""
        (
            self._nodes,
            self._node_rows,
            self._node_adjacencies,
            self._node_adj_probs,
            self._node_degrees,
        ) = utils.get_combo_nodes(
            self._combos,
            self._adjacencies,
            self._adjacency_probs,
            self._degrees_lookup.value(),
        )

    # noinspection PyUnreachableCode
    def _get_predictions(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Obtains the model prediction on current neighbour node combinations.."""
        current_batch = utils.get_combo_inputs(
            tf.squeeze(self.skel_img),
            tf.squeeze(self.node_pos),
            self._combos,
            self._pos_list_xy,
        )
        probabilities = self._predict_func(*current_batch)

        adj, adj_probs = utils.classify(probabilities)

        # expand dimensions of (adj, adj_probs) if their lengths are one
        return tf.cond(
            tf.size(adj) > 1,
            true_fn=lambda: (adj, adj_probs),
            false_fn=lambda: (
                tf.expand_dims(adj, axis=-1),
                tf.expand_dims(adj_probs, axis=-1),
            ),
        )

    def _predict_ok_good(self) -> None:
        """Generates list of combos and nodes found that match the case
            NODE_ADJ <= NODE_DEGREES.
        Then updates the lookup tables and remove the found combinations and nodes
        from the placeholder lists of nodes and combinations.
        """
        (
            case_combos,
            case_adj,
            self._nodes,
            self._node_rows,
            self._node_adjacencies,
        ) = utils.predict_ok_good(
            self._nodes,
            self._node_rows,
            self._node_adjacencies,
            self._node_adj_probs,
            self._node_degrees,
            self._combos,
            self._adjacencies,
            self._degrees_lookup.value(),
        )
        self._update_after_prediction(case_combos, case_adj)

    def _predict_bad(self) -> None:
        """Generates list of combos and nodes found that match the case
            NODE_ADJ > NODE_DEGREES.
        Chooses the nodes and combinations such that the NEW_NODE_ADJ -  NODE_DEGREES >= 0.
        Then updates the lookup tables and remove the found combinations and nodes
        from the placeholder lists of nodes and combinations.
        """
        (
            case_combos,
            case_adj,
            self._nodes,
            self._node_rows,
            self._node_adjacencies,
        ) = utils.predict_bad(
            self._nodes,
            self._node_rows,
            self._node_adjacencies,
            self._node_adj_probs,
            self._node_degrees,
            self._combos,
            self._degrees_lookup.value(),
        )
        self._update_after_prediction(case_combos, case_adj)

    def _update_after_prediction(
        self, case_combos: tf.Tensor, case_adj: tf.Tensor
    ) -> None:
        """Updates adjacency matrix, then removes the combinations and nodes recently found
        from the lists of node pair combinations."""
        utils.update_lookup(
            self._degrees_lookup,
            self._nodes,
            self._node_adjacencies,
        )

        # update adjacency matrix, remove found combinations and nodes
        tf.cond(
            tf.size(case_combos) > 0,
            true_fn=lambda: self._update_A_and_combos_from_case(case_combos, case_adj),
            false_fn=lambda: None,
        )

        # set stop flag if pool of possible combos has reduced to zero
        tf.cond(
            tf.size(self._reduced_combos) == 0,
            true_fn=lambda: self._set_stop(),
            false_fn=lambda: None,
        )

        # update list of nodes from the remaining combos
        self._update_nodes()

    def _update_A_and_combos_from_case(
        self, case_combos: tf.Tensor, case_adj: tf.Tensor
    ) -> None:
        # update adjacency matrix
        self._update_func(case_combos, case_adj, self._A)

        # update combos: remove found combinations
        self._reduced_combos = utils.remove_combo_subset_from_all(
            self._reduced_combos, case_combos
        )
        (
            self._combos,
            self._adjacencies,
            self._adjacency_probs,
        ) = utils.remove_combo_subset(
            self._combos, self._adjacencies, self._adjacency_probs, case_combos
        )

        # update combos: remove found nodes
        nodes_found = utils.nodes_found(self._degrees_lookup.value())
        tf.cond(
            tf.size(nodes_found) > 0,
            true_fn=lambda: self._remove_found_nodes(nodes_found),
            false_fn=lambda: None,
        )

    def _remove_found_nodes(self, nodes_found: tf.Tensor) -> None:
        (
            self._reduced_combos,
            self._combos,
            self._adjacencies,
            self._adjacency_probs,
        ) = utils.remove_nodes_found(
            self._reduced_combos,
            self._combos,
            self._adjacencies,
            self._adjacency_probs,
            nodes_found,
        )

    def _set_stop(self) -> None:
        """Sets flag to stop iterating."""
        self._stop_iterate = tf.constant(True)

    @property
    def nodes_not_found(self) -> tf.Tensor:
        return nodes_not_found(self._degrees_lookup.value())

    @property
    def num_combos(self) -> tf.Tensor:
        return tf.shape(self._combos)[0]

    # parameters
    @property
    def algorithm(self) -> AdjCombinationSchemes:
        return AdjCombinationSchemes(self._data_container.algorithm)

    @property
    def k0(self) -> tf.Tensor:
        return tf.constant(self._data_container.num_neighbours)

    # inputs
    @property
    def skel_img(self) -> tf.Tensor:
        return self._data_container.skel_image_tensor

    @property
    def node_pos(self) -> tf.Tensor:
        return self._data_container.node_pos_tensor

    @property
    def node_degrees(self) -> tf.Tensor:
        return self._data_container.node_deg_tensor

    # outputs
    @property
    def A(self) -> np.ndarray:
        return self._A.value().numpy()

    @property
    def pos_list_xy(self) -> np.ndarray:
        return self._pos_list_xy.numpy()
