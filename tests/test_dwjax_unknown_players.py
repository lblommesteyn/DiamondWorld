import jax.numpy as jnp

from diamondworldjax.model.embeddings import PlayerRegistry
from diamondworldjax.scripts.train_pa import _map_player_ids


def test_unseen_players_and_padding_do_not_alias_first_real_player():
    batch = {
        "pitcher_ids": jnp.array([[10, 999, 0]], dtype=jnp.int32),
        "batter_ids": jnp.array([[20, 888, 0]], dtype=jnp.int32),
    }
    mapped = _map_player_ids(batch, {10: 0, 20: 1})
    assert mapped["pitcher_ids"].tolist() == [[0, 2, 2]]
    assert mapped["batter_ids"].tolist() == [[1, 2, 2]]


def test_registry_returns_neutral_embedding_for_sentinel():
    # Construct only the lookup wrapper; Flax module initialisation is irrelevant
    # to the sentinel behaviour under test.
    registry = PlayerRegistry.__new__(PlayerRegistry)
    registry._table = jnp.array([[3.0, 4.0], [5.0, 6.0]])

    out = registry.lookup(jnp.array([0, 1, 2, -1], dtype=jnp.int32))
    assert out.tolist() == [[3.0, 4.0], [5.0, 6.0], [0.0, 0.0], [0.0, 0.0]]
