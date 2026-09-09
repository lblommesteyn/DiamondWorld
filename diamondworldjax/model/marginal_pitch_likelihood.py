"""Observed-data ABCD likelihood with current-pitch measurement marginalization.

Discrete pitch types are enumerated exactly; continuous missing stuff/launch are
integrated with reparameterized Monte Carlo. Historical missing measurements use
an explicitly mask-aware encoder, not retrospective or future-data imputation.
"""
import jax
import jax.numpy as jnp
import optax


def _categorical(logits, target):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), target[..., None].astype(jnp.int32), -1)[..., 0]


def _normal(y, mu, ls, observed):
    safe_y = jnp.where(observed, y, mu)
    return jnp.where(observed, -.5 * ((safe_y - mu) * jnp.exp(-ls))**2 - ls
                     - .5 * jnp.log(2 * jnp.pi), 0.).sum(-1)


def marginal_log_likelihood(apply, batch, key, samples=2):
    """Sum per-target log marginals; no extra loss for warm-up history.

    `apply(batch)` returns a dict keyed by the selected ABCD heads, including A.
    Samples affect only current conditioning fields. History values/masks stay
    fixed across integration draws, so a draw for pitch t cannot leak into t-1.
    """
    if samples < 1:
        raise ValueError('Marginal likelihood requires samples >= 1')
    b = dict(batch)
    sm = b.get('stuff_observed', jnp.broadcast_to(b['stuff_valid'][..., None] > 0, b['stuff'].shape))
    lm = b.get('launch_observed', jnp.broadcast_to(b['launch_valid'][..., None] > 0, b['launch'].shape))
    tm = b['type_valid'] > 0
    b['stuff'] = jnp.where(sm, b['stuff'], 0.)
    b['launch'] = jnp.where(lm, b['launch'], 0.)
    b['pitch_type'] = jnp.where(tm, b['pitch_type'], 0)
    b['stuff_observed'], b['launch_observed'] = sm, lm
    b['history_stuff'], b['history_type'] = b['stuff'], b['pitch_type']
    apply = jax.checkpoint(apply)
    a = apply(b)['a']
    log_type = jax.nn.log_softmax(a['type_logits'])
    component_log = _normal(b['stuff'][..., None, :], a['stuff_mu'], a['stuff_logsigma'], sm[..., None, :])
    type_index = jnp.arange(8)
    allowed = (~tm[..., None]) | (b['pitch_type'][..., None] == type_index)
    mixture_log = jnp.where(allowed, log_type + component_log, -jnp.inf)
    base_log = jax.scipy.special.logsumexp(mixture_log, -1)
    conditional_type_log = mixture_log - base_log[..., None]
    launch_eligible = b['batted_valid'] > 0
    # A measured launch can be scored even if its outcome label is unavailable.
    launch_eligible = launch_eligible | lm.any(-1)

    def draw(index):
        kind = index // samples
        rng = jax.random.fold_in(key, index)
        stuff_key, launch_key = jax.random.split(rng)
        mu, ls = a['stuff_mu'][..., kind, :], a['stuff_logsigma'][..., kind, :]
        latent_stuff = mu + jnp.exp(ls) * jax.random.normal(stuff_key, mu.shape)
        current = {**b, 'pitch_type': jnp.full_like(b['pitch_type'], kind),
                   'stuff': jnp.where(sm, b['stuff'], latent_stuff)}
        out = apply(current)
        downstream = jnp.zeros_like(base_log)
        if 'b' in out:
            for label, eligible in [('swing', jnp.ones_like(tm)), ('contact', b['swing'] > 0),
                    ('foul', (b['swing'] > 0) & (b['contact'] > 0)), ('hbp', b['swing'] == 0)]:
                lp = -optax.sigmoid_binary_cross_entropy(out['b'][label + '_logit'], b[label])
                downstream += jnp.where(eligible, lp, 0.)
        if 'c' in out:
            logits = out['c']['event_logits']
            if logits.shape[-1] == 256:
                target = (b['events'].astype(jnp.int32) * (1 << jnp.arange(8))).sum(-1)
                lp = _categorical(logits, target)
            else:
                lp = -optax.sigmoid_binary_cross_entropy(logits, b['events']).sum(-1)
            eligible = b.get('c_eligible', ~b.get('pa_terminal', jnp.zeros_like(tm)))
            downstream += jnp.where(eligible, lp, 0.)
        if 'd' in out:
            d = out['d']
            downstream += jnp.where(launch_eligible,
                _normal(b['launch'], d['launch_mu'], d['launch_logsigma'], lm), 0.)
            latent_launch = d['launch_mu'] + jnp.exp(d['launch_logsigma']) * jax.random.normal(launch_key, b['launch'].shape)
            # Marginalize missing launch; never condition D2 on storage fills.
            launch = jnp.where(lm, b['launch'], latent_launch)
            outcome = apply({**current, 'launch': launch})['d']['outcome_logits']
            downstream += jnp.where(b['batted_valid'] > 0, _categorical(outcome, b['batted_out']), 0.)
        return downstream + conditional_type_log[..., kind]

    draws = jax.lax.map(draw, jnp.arange(8 * samples))
    marginal = base_log + jax.scipy.special.logsumexp(draws, axis=0) - jnp.log(float(samples))
    target = b['valid'] * b.get('loss_mask', 1)
    return jnp.where(target > 0, marginal, 0.).sum()
