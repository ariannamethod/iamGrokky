"""Training utilities for SLNCX.

This module provides helpers to prepare data, run a minimal JAX-based
training loop and manage checkpoints. It also contains lower level
functions used by :mod:`SLNCX.model` to restore parameters.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import pickle
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

rank_logger = logging.getLogger("rank")


@contextlib.contextmanager
def copy_to_shm(file: str):
    if file.startswith("/dev/shm/"):
        # Nothing to do, the file is already in shared memory.
        yield file
        return

    tmp_dir = "/dev/shm/"
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir)
    try:
        shutil.copyfile(file, tmp_path)
        yield tmp_path
    finally:
        os.remove(tmp_path)
        os.close(fd)


def fast_unpickle(path: str) -> Any:
    with copy_to_shm(path) as tmp_path:
        with open(tmp_path, "rb") as f:
            return pickle.load(f)


def load_tensors(shaped_arrays, directory, mesh_config, tensor_indices=None):
    """Load tensors from ``directory``.

    Parameters
    ----------
    shaped_arrays:
        Sequence describing the arrays to load.
    directory: str
        Directory containing serialized tensors.
    mesh_config: tuple
        Mesh layout used to determine which host loads which tensor.
    tensor_indices: iterable | None
        Optional subset of tensor indices to load.
    """
    pool = ThreadPoolExecutor(max_workers=32)
    fs = []
    num_replicas = 1
    data_model_shards = math.prod(mesh_config)
    if tensor_indices is None:
        iterator = enumerate(shaped_arrays)
    else:
        iterator = zip(tensor_indices, shaped_arrays)
    for i, t in iterator:
        if (i % num_replicas) == (
            (jax.process_index() // data_model_shards) % num_replicas
        ):
            idx = (
                jax.process_index()
                // (num_replicas * data_model_shards)
                * data_model_shards
                + jax.process_index() % data_model_shards
            )
            fs.append(
                pool.submit(
                    fast_unpickle, os.path.join(directory, f"tensor{i:05d}_{idx:03d}")
                )
            )
        else:
            fs.append(pool.submit(np.zeros, t.shape, dtype=t.dtype))
    wait(fs)
    return [f.result() for f in fs]


def path_tuple_to_string(path: tuple) -> str:
    pieces = []
    for elem in path:
        if isinstance(elem, jax.tree_util.DictKey):
            pieces.append(elem.key)
        elif isinstance(elem, jax.tree_util.GetAttrKey):
            pieces.append(elem.name)
        else:
            assert isinstance(
                elem, (jax.tree_util.FlattenedIndexKey, jax.tree_util.SequenceKey)
            )
    return "/".join(pieces)


def get_load_path_str(
    init_path_str: str,
    load_rename_rules: Optional[list[tuple[str, str]]] = None,
    load_exclude_rules: Optional[list[str]] = None,
) -> Optional[str]:
    """Apply renaming and exclusion rules to ``init_path_str``."""
    if load_exclude_rules is not None:
        for search_pattern in load_exclude_rules:
            if re.search(search_pattern, init_path_str):
                return None

    load_path_str = init_path_str
    if load_rename_rules is not None:
        for search_pattern, replacement_pattern in load_rename_rules:
            if re.search(search_pattern, load_path_str):
                load_path_str = re.sub(
                    search_pattern, replacement_pattern, load_path_str
                )
                break

    return load_path_str


def restore(
    *,
    checkpoint_path: str,
    state_shapes: Any,
    mesh,
    between_hosts_config,
    params_only: bool,
    state_sharding,
    init_state: Optional[Any] = None,
) -> Any:
    """Restore parameters from a checkpoint."""
    ckpt_path = os.path.join(checkpoint_path, "ckpt-0")
    rank_logger.info("Loading checkpoint at %s", ckpt_path)
    ckpt_shapes = state_shapes
    ckpt_shapes_with_path, structure = jax.tree_util.tree_flatten_with_path(ckpt_shapes)
    ckpt_shapes_flat = [elem[1] for elem in ckpt_shapes_with_path]
    loaded_tensors = load_tensors(ckpt_shapes_flat, ckpt_path, between_hosts_config)
    state = jax.tree_util.tree_unflatten(structure, loaded_tensors)

    ckpt_keys = set(state.params.keys())
    code_keys = set(state_sharding.params.keys())
    if ckpt_keys != code_keys and init_state is None:
        missing_in_ckpt = code_keys - ckpt_keys
        missing_locally = ckpt_keys - code_keys
        raise ValueError(
            "Parameters in the code are not matching checkpoint parameters.\n"
            "Params missing in checkpoint: {}\nParams missing in code: {}".format(
                missing_in_ckpt, missing_locally
            )
        )
    state_sharding = jax.tree_util.tree_map(
        lambda x: jax.sharding.PartitionSpec() if x is None else x,
        state_sharding,
        is_leaf=lambda x: x is None,
    )
    state = multihost_utils.host_local_array_to_global_array(
        state, mesh, state_sharding
    )
    if params_only:
        state = state.params
    return state


# ---------------------------------------------------------------------------
# High level training helpers
# ---------------------------------------------------------------------------


def prepare_data(path: str) -> jnp.ndarray:
    """Load a text file and return an array of token ids."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return jnp.array([ord(c) for c in text], dtype=jnp.int32)


def train(
    init_fn: Callable[[jax.Array, jnp.ndarray], Any],
    apply_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    data: jnp.ndarray,
    *,
    num_steps: int = 100,
    learning_rate: float = 1e-3,
) -> Any:
    """Run a simple gradient-descent training loop."""
    rng = jax.random.PRNGKey(0)
    params = init_fn(rng, data)

    def loss_fn(p):
        preds = apply_fn(p, data)
        return jnp.mean((preds - data) ** 2)

    for _ in range(num_steps):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params


def save_checkpoint(params: Any, path: str) -> None:
    """Save ``params`` to ``path`` using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(params, f)
