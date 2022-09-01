from typing import Tuple

import numpy as np
import tensorflow as tf


@tf.function
def unsorted_pos_list_from_image(node_pos_img: tf.Tensor) -> tf.Tensor:
    """Extracts the (unsorted) xy coordinates from the node_pos image."""
    node_pos_img = tf.cast(tf.squeeze(node_pos_img), tf.uint8)
    return tf.reverse(tf.where(node_pos_img), axis=[1])


@tf.function
def get_sort_indices(xy_unsorted: tf.Tensor) -> tf.Tensor:
    """
    Returns the sort indices when sorting by y/rows first, then x/cols.
    """
    idx_y_sorted = tf.argsort(xy_unsorted[:, 1])
    y_sorted = tf.gather(xy_unsorted, idx_y_sorted)

    idx_x_sorted = tf.argsort(y_sorted[:, 0])
    idx_xy_sorted = tf.gather(idx_y_sorted, idx_x_sorted)

    return idx_xy_sorted


@tf.function
def sorted_pos_list_from_image(node_pos_img: tf.Tensor) -> tf.Tensor:
    """Extracts the sorted xy coordinates from the node_pos image."""
    xy_unsorted = unsorted_pos_list_from_image(node_pos_img)
    sort_indices = get_sort_indices(xy_unsorted)
    return tf.gather(xy_unsorted, sort_indices)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.uint8),
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.uint8),
    ]
)
def data_from_node_imgs(
    node_pos: tf.Tensor, degrees: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Returns lookup table consisting of node xy position and the corresponding degree."""
    pos_list_xy = sorted_pos_list_from_image(node_pos)
    pos_list_rc = tf.reverse(pos_list_xy, axis=[1])
    degrees_list = tf.gather_nd(indices=pos_list_rc, params=tf.squeeze(degrees))
    num_nodes = tf.shape(pos_list_xy, out_type=tf.int64)[0]

    return pos_list_xy, degrees_list, num_nodes


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int64)])
def get_all_node_combinations(num_nodes: tf.Tensor) -> tf.Tensor:
    indices = tf.range(num_nodes)

    # meshgrid to create all possible pair combinations
    a, b = tf.meshgrid(indices, indices)
    a = tf.linalg.band_part(a, 0, -1)
    b = tf.linalg.band_part(b, 0, -1)
    grid = tf.stack((a, b), axis=-1)
    combos = tf.reshape(grid, shape=(num_nodes * num_nodes, 2))

    # remove pairs where both nodes are the same
    idcs_not_equal = tf.where(tf.not_equal(combos[:, 0], combos[:, 1]))
    combos = tf.gather_nd(combos, idcs_not_equal)

    return combos


# noinspection PyPep8Naming
@tf.function
def _update(combos: tf.Tensor, adjacencies: tf.Tensor, A: tf.Variable):
    A.scatter_nd_update(combos, adjacencies)
    A.scatter_nd_update(tf.reverse(combos, axis=[-1]), adjacencies)


# noinspection PyPep8Naming
def get_update_function(A: tf.Variable) -> tf.types.experimental.ConcreteFunction:
    return _update.get_concrete_function(
        combos=tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        adjacencies=tf.TensorSpec(shape=(None,), dtype=tf.int64),
        A=A,
    )


def get_placeholders(
    degrees_list: tf.Tensor, num_nodes: tf.Tensor
) -> Tuple[tf.Variable, tf.Variable]:
    degrees_var = tf.Variable(initial_value=degrees_list, trainable=False)
    A = tf.Variable(
        initial_value=tf.zeros((num_nodes, num_nodes), dtype=tf.int64),
        trainable=False,
    )

    return degrees_var, A


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(256, 256), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256), dtype=tf.uint8),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    ],
)
def get_combo_inputs(
    skel_img: tf.Tensor,
    node_pos_img: tf.Tensor,
    combos: tf.Tensor,
    pos_list_xy: tf.Tensor,
) -> tf.data.Dataset:
    img_dims = tf.shape(skel_img)
    combos_xy = tf.gather(pos_list_xy, combos)
    combo_imgs = get_combo_imgs_from_xy(combos_xy, img_dims)
    num_neighbours = tf.cast(tf.shape(combos_xy)[0], tf.int64)

    return (
        combo_imgs.map(lambda x: (skel_img, node_pos_img, x))
        .batch(num_neighbours)
        .get_single_element()
    )


def get_combo_imgs_from_xy(
    combos_xy: tf.Tensor,
    img_dims: tf.Tensor,
) -> tf.data.Dataset:
    combos_rc = tf.reverse(combos_xy, axis=[-1])
    ds = tf.data.Dataset.from_tensor_slices(combos_rc)
    return ds.map(
        lambda x: tf.numpy_function(
            rc_to_node_combo_img, inp=(x[0], x[1], img_dims), Tout=tf.int64
        )
    )


def rc_to_node_combo_img(
    rc1: np.ndarray, rc2: np.ndarray, dims: np.ndarray
) -> np.ndarray:
    """ "Converts a pair of (row, col) coordinates to a blank image
    with white dots corresponding to the given coordinates."""
    img = np.zeros(dims, dtype=np.int64)

    img[rc1[0], rc1[1]] = 1
    img[rc2[0], rc2[1]] = 1

    return img


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32)])
def tf_classify(output: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.greater(output, tf.constant(0.5)), tf.int64)


@tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32)])
def classify(probs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    adjacency_probs = tf.squeeze(probs)
    adjacencies = tf_classify(adjacency_probs)
    return adjacency_probs, adjacencies


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ]
)
def get_combo_nodes(
    combos: tf.Tensor,
    adjacencies: tf.Tensor,
    adjacency_probs: tf.Tensor,
    degrees_list: tf.Tensor,
):
    """Returns only the nodes (+ other node data) that are in the given node pair combinations."""
    nodes, node_rows = unique_nodes_from_combo(combos)
    node_adj, node_adj_probs = node_adjacencies(node_rows, adjacencies, adjacency_probs)
    node_degrees = tf.cast(tf.gather(degrees_list, nodes), tf.int64)

    return nodes, node_rows, node_adj, node_adj_probs, node_degrees


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    ],
)
def unique_nodes_from_combo(combos: tf.Tensor) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    """Returns a list of nodes present in the given node pair combinations without duplicates,
    as well as the corresponding indices relative to the list of node pair combinations."""
    # flatten combos
    num_elems_combos = tf.reduce_prod(
        tf.shape(combos, out_type=tf.int64), keepdims=True
    )
    nodes = tf.unique(tf.reshape(combos, num_elems_combos)).y

    # get row indices for each unique node
    rows_in_combos = tf.map_fn(
        lambda node: tf.where(combos == node)[:, 0],
        elems=nodes,
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int64),
    )

    return nodes, rows_in_combos


@tf.function(
    input_signature=[
        tf.RaggedTensorSpec(shape=None, dtype=tf.int64, ragged_rank=1),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ],
)
def node_adjacencies(
    rows_in_combos: tf.RaggedTensor, adjacencies: tf.Tensor, adjacency_probs: tf.Tensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    """Returns the corresponding adjacencies for nodes given their indices relative to
    a list of node pair combinations."""
    unique_adjacencies = tf.reduce_sum(tf.gather(adjacencies, rows_in_combos), axis=1)
    unique_adj_probs = tf.gather(adjacency_probs, rows_in_combos)

    return unique_adjacencies, unique_adj_probs
