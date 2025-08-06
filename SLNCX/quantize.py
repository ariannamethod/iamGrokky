# Minimal script to quantize checkpoint weights to 2-bit.
# Usage: python quantize.py <checkpoint_dir> <output_dir>

import argparse
import jax
import jax.numpy as jnp

from model import (
    ModelRunner,
    QuantizedWeight2bit,
    default_config,
    fast_pickle,
    restore,
)


def quantize_tensor(tensor: jax.Array) -> QuantizedWeight2bit:
    """Symmetric per-tensor quantization to 2 bits."""
    scale = jnp.maximum(jnp.max(jnp.abs(tensor)) / 1.0, 1e-8)
    q_weight = jnp.clip(jnp.round(tensor / scale), -2, 1).astype(jnp.int8)
    return QuantizedWeight2bit(weight=q_weight, scales=scale.astype(jnp.float32))


def dequantize_tensor(qw: QuantizedWeight2bit) -> jax.Array:
    """Dequantize a :class:`QuantizedWeight2bit` back to float32."""
    return qw.weight.astype(jnp.float32) * qw.scales


def quantize_params(params):
    def _quantize(x):
        if x.dtype in (jnp.float32, jnp.bfloat16):
            return quantize_tensor(jnp.asarray(x, dtype=jnp.float32))
        return x

    return jax.tree_util.tree_map(_quantize, params)


def main(args):
    config = default_config()
    config.initialize()

    dummy_data = {
        "inputs": jnp.zeros((1, 1), dtype=jnp.int32),
        "targets": jnp.zeros((1, 1), dtype=jnp.int32),
    }
    runner = ModelRunner(
        model=config, bs_per_device=0.125, checkpoint_path=args.checkpoint
    )
    runner.transform_forward = True
    runner.initialize(dummy_data, local_mesh_config=(1, 1), between_hosts_config=(1, 1))
    state_shapes = jax.eval_shape(runner.init_fn, jax.random.PRNGKey(0), dummy_data)
    params = restore(
        checkpoint_path=args.checkpoint,
        state_shapes=state_shapes,
        mesh=runner.mesh,
        between_hosts_config=(1, 1),
        params_only=True,
        state_sharding=runner.state_sharding,
        init_state=None,
    )
    q_params = quantize_params(params)
    fast_pickle(q_params, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize checkpoint weights to 2-bit")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument("output", help="Output path for quantized weights")
    main(parser.parse_args())
