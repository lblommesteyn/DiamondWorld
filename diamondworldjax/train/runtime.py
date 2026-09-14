"""Bounded batch preparation and shape-stable compiled training utilities."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp


def prefetch(iterable, depth=2):
    """Prepare at most depth future batches on one worker, preserving order/errors."""
    if depth < 1:
        yield from iterable
        return
    iterator, end = iter(iterable), object()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dwjax-batches")
    pending = deque()
    try:
        pending.extend(pool.submit(next, iterator, end) for _ in range(depth))
        while pending:
            value = pending.popleft().result()
            if value is end:
                return
            pending.append(pool.submit(next, iterator, end))
            yield value
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def tree_finite(tree):
    leaves = jax.tree.leaves(tree)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves])) if leaves else jnp.array(True)


def safe_update(update, state, *args, **kwargs):
    """Rollback optimizer and parameters together on a nonfinite loss or state."""
    candidate, loss = update(state, *args, **kwargs)
    finite = jnp.isfinite(loss) & tree_finite(candidate)
    state = jax.lax.cond(finite, lambda: candidate, lambda: state)
    return state, jnp.where(finite, loss, jnp.nan)


def shape_signature(tree):
    leaves, structure = jax.tree.flatten(tree)
    return structure, tuple((x.shape, x.dtype) for x in leaves)


def stack_batches(batches):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *batches)


def length_bucket(length, minimum=32):
    """Power-of-two lengths limit compilation variants without dropping targets."""
    return max(minimum, 1 << max(0, int(length) - 1).bit_length())


def bucket_batch(batch, valid_key="valid"):
    """Trim trailing padding only, retaining left context and absolute positions."""
    import numpy as np
    valid = np.asarray(batch[valid_key])
    if valid.ndim < 2:
        return batch
    used = np.flatnonzero(valid.any(axis=tuple(range(valid.ndim-1))))
    size = min(valid.shape[-1], length_bucket(int(used[-1])+1 if used.size else 1))
    slices = (slice(None),) * (valid.ndim-1) + (slice(size),)
    return {k: v[slices] if getattr(v, 'ndim', 0) >= valid.ndim
            and v.shape[:valid.ndim] == valid.shape else v for k, v in batch.items()}
