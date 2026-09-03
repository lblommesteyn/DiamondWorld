import numpy as np

from diamondworldjax.sim.game_evaluation import simulate_score_draws


def test_generated_game_evaluation_preserves_every_mc_draw(monkeypatch):
    import diamondworldjax.scripts.simulate_games as simulator

    def fake_simulate(model_fn, params, player_table, games, rng_key, **kwargs):
        del model_fn, params, player_table, rng_key, kwargs
        # Inputs are game-major: g0/s0, g0/s1, g1/s0, g1/s1.
        return {
            "away": np.array([1, 2, 3, 4]),
            "home": np.array([5, 6, 7, 8]),
            "away_hits": np.zeros(4),
            "home_hits": np.zeros(4),
        }

    monkeypatch.setattr(simulator, "simulate", fake_simulate)
    away, home = simulate_score_draws(None, None, {}, [{"game_pk": 1}, {"game_pk": 2}],
                                      None, num_samples=2)
    assert away.tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert home.tolist() == [[5.0, 7.0], [6.0, 8.0]]
