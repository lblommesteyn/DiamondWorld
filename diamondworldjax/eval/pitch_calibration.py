"""Held-out pitch-resolution scoring, distinct from PA-start prediction."""
from __future__ import annotations
import numpy as np


def pitch_resolution_probs(swing, contact, foul, hbp, d_probs, called_strike, balls, strikes):
    """Nine terminal outcomes plus continuation, conditional on a realized pitch."""
    s, c, f, h = [np.asarray(v, dtype=float) for v in (swing, contact, foul, hbp)]
    cs = np.asarray(called_strike, dtype=float)
    out = np.zeros((*s.shape, 10), dtype=float)
    take = 1 - s
    out[..., 2] = take * h
    out[..., 0] = (s * (1 - c) + take * (1 - h) * cs) * (strikes >= 2)
    out[..., 1] = take * (1 - h) * (1 - cs) * (balls >= 3)
    bip = s * c * (1 - f)
    out[..., [7, 3, 4, 5, 6, 8]] = bip[..., None] * d_probs
    out[..., 9] = np.maximum(1 - out[..., :9].sum(-1), 0)
    return out


def score_heldout(heads, arrays, batch_size=16, seed=0, launch_samples=16):
    """Score observed outcomes, integrating D1 launch rather than using observed launch.

    Conditions on the observed pitch and pre-pitch history. This is not a
    PA-start probability, and does not score A's pitch density.
    """
    import jax
    import jax.numpy as jnp
    from diamondworldjax.data.pitch_seq import STUFF_CENTRE, STUFF_SCALE
    if heads.d is None:
        raise ValueError("Held-out resolution calibration requires D")
    key = jax.random.PRNGKey(seed)
    probs, labels = [], []
    @jax.jit
    def predict(batch, key):
        b = heads.b.apply(heads.b_params, batch, train=False)
        d = heads.d.apply(heads.d_params, batch, train=False)
        keys = jax.random.split(key, launch_samples)
        def sample(k):
            launch = d["launch_mu"] + jnp.exp(d["launch_logsigma"]) * jax.random.normal(k, d["launch_mu"].shape)
            result = heads.d.apply(heads.d_params, {**batch, "launch": launch}, train=False)
            return jax.nn.softmax(result["outcome_logits"])
        dp = jax.lax.map(sample, keys).mean(0)
        cs = (jax.nn.sigmoid(b["called_strike_logit"])
              if "called_strike_logit" in b else None)
        return [jax.nn.sigmoid(b[name + "_logit"]) for name in ("swing", "contact", "foul", "hbp")], cs, dp
    n = len(arrays["valid"])
    for start in range(0, n, batch_size):
        batch = {k: jnp.asarray(v[start:start + batch_size]) for k, v in arrays.items()}
        key, subkey = jax.random.split(key)
        bp, cs, dp = predict(batch, subkey)
        stuff = np.asarray(batch["stuff"])
        px = stuff[..., 3] * STUFF_SCALE[3] + STUFF_CENTRE[3]
        pz = stuff[..., 4] * STUFF_SCALE[4] + STUFF_CENTRE[4]
        zone = (np.abs(px) <= .83) & (pz >= 1.52) & (pz <= 3.42)
        ctx = np.asarray(batch["ctx"])
        cs = zone if cs is None else np.asarray(cs)
        p = pitch_resolution_probs(*map(np.asarray, bp), np.asarray(dp), cs,
                                   np.rint(ctx[..., 0] * 3), np.rint(ctx[..., 1] * 2))
        observed = np.asarray(batch["pa_outcome"])
        terminal = np.asarray(batch["pa_terminal"], bool)
        target = np.where(terminal, observed, 9)
        valid = (np.asarray(batch["valid"] * batch.get("loss_mask", 1), bool) & (target >= 0)
                 & (np.asarray(batch["stuff_valid"]) > 0))
        probs.append(p[valid]); labels.append(target[valid])
    p, y = np.concatenate(probs), np.concatenate(labels)
    if not len(y):
        return {"n": 0, "scope": "heldout_next_pitch_resolution"}
    chosen = p[np.arange(len(y)), y]
    low = p.cumsum(-1)[np.arange(len(y)), y] - chosen
    pit = low + np.random.default_rng(seed).random(len(y)) * chosen
    return {"scope": "heldout_next_pitch_resolution", "n": len(y),
            "conditioning": "observed pitch and prior history; launch marginalized; includes continuation",
            "nll": float(-np.log(np.maximum(chosen, 1e-12)).mean()),
            "brier": float(((p - np.eye(10)[y]) ** 2).sum(-1).mean()),
            "pit_histogram": np.histogram(pit, bins=np.linspace(0, 1, 11))[0].tolist(),
            "predicted_rates": p.mean(0).tolist(),
            "observed_rates": (np.bincount(y, minlength=10) / len(y)).tolist(),
            "launch_samples": launch_samples}
