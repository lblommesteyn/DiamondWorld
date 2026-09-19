"""Native variational ABCD skills. Transformer weights are point estimates.

A world draw samples the shared latent once and each head residual once. Exported
mean tables retain the existing standalone transformer checkpoint interface.
"""
from copy import deepcopy
import pickle
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from .embeddings import PlayerSeasonEncoder, SkillFusionLayer, SKILL_DIM
from .pitchformer import SuperState, TransformerA, TransformerB
from .transformer_c import TransformerC
from .transformer_d import TransformerD


class BayesianNetwork(nn.Module):
    options: dict
    heads: str
    f_player: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, batch, features, skills, role_indices, *, train=False,
                 hidden_override=None, encode_only=False, selected_heads=None):
        if hidden_override is not None:
            outputs = {}
            for head in (selected_heads or self.heads):
                cls = dict(a=TransformerA, b=TransformerB, c=TransformerC, d=TransformerD)[head]
                outputs[head] = cls(**self.options, dropout=self.dropout, name='head_' + head)(
                    batch, train=train, hidden_override=hidden_override[head])
            return outputs, {}
        # The same encoder/fusion architecture and latent width used by PA.
        stats, league, hand = features
        # v26 and earlier stored one player table, whose outcome-rate columns
        # described batting only. Accept those checkpoints, but new training
        # provides separate pitcher and batter tables so a pitcher's encoder
        # can see the results he allowed rather than an all-zero rate vector.
        # The two role encoders share weights; role-specific inputs are enough
        # to preserve a common skill space without conflating their evidence.
        if stats.ndim == 2:
            stats = jnp.broadcast_to(stats, (2, *stats.shape))
            league = jnp.broadcast_to(league, (2, *league.shape))
            hand = jnp.broadcast_to(hand, (2, *hand.shape))
        encoder = PlayerSeasonEncoder(self.f_player, name="encoder")
        encoded = [encoder(stats[i], league[i], hand[i]) for i in range(2)]
        fusion = SkillFusionLayer(name="fusion")
        ss_module = SuperState(**{k: v for k, v in self.options.items()
                                  if k in ('n_pitchers', 'n_batters', 'n_parks', 'd_model',
                                           'player_mode', 'skill_seasons')}, name="shared_ss")
        outputs = {}
        tables = {}
        for i, head in enumerate(self.heads):
            skill = skills[0] + skills[i + 1]
            tables[head] = {}
            vectors = []
            for role_i, role in enumerate(('pitcher', 'batter')):
                det = jnp.broadcast_to(encoded[role_i][:, None, :],
                                       (*skill.shape[:2], 64))
                table = fusion(det, skill)
                # Unknown sentinel is never a learned player.
                table = table.at[0].set(0.)
                tables[head][role] = table
                ids = batch[role + '_idx']
                mapping = role_indices[role]
                known = (ids > 0) & (ids < len(mapping))
                universal = mapping[jnp.clip(ids, 0, len(mapping) - 1)]
                vectors.append(jnp.where(known[..., None],
                    table[universal, batch['skill_season']], 0.))
            ss = ss_module(batch['pitcher_idx'], batch['batter_idx'], batch['park_idx'],
                           batch['ctx'], batch['geom'], player_vectors=vectors)
            cls = dict(a=TransformerA, b=TransformerB, c=TransformerC, d=TransformerD)[head]
            outputs[head] = cls(**self.options, dropout=self.dropout, name='head_' + head)(
                batch, train=train, ss_override=ss, encode_only=encode_only)
        return outputs, tables


def draw_skills(posterior, key, residual_scale, prior, walk_scale, sample=True):
    """Joint draw in innovation space; row zero has no posterior or KL cost."""
    mu, rho = posterior['mu'], posterior['rho']
    x = mu + jax.nn.softplus(rho) * jax.random.normal(key, mu.shape) if sample else mu
    if prior == 'walk':
        x = jnp.concatenate([x[:, :, :1], x[:, :, 1:] * walk_scale], axis=2)
        x = jnp.cumsum(x, axis=2)
    return jnp.pad(x, ((0, 0), (1, 0), (0, 0), (0, 0)))


def skill_kl(posterior, residual_scale):
    sigma = jax.nn.softplus(posterior['rho'])
    scales = jnp.asarray([np.sqrt(1 - residual_scale ** 2)] +
                         [residual_scale] * (sigma.shape[0] - 1))[:, None, None, None]
    return jnp.sum(jnp.log(scales / sigma) +
                   (sigma ** 2 + posterior['mu'] ** 2) / (2 * scales ** 2) - .5)


def likelihood(outputs, b, d_hr_weight: float = 0.0):
    """Summed conditional likelihood, with missing conditioning data excluded."""
    total = 0.
    if d_hr_weight < 0:
        raise ValueError('d_hr_weight must be nonnegative')
    v = (b['valid'] * b.get('loss_mask', 1)).astype(bool)
    tracked = v & b['type_valid'].astype(bool) & b['stuff_valid'].astype(bool)
    def add(lp, mask):
        return jnp.where(mask, lp, 0.).sum()
    def categorical(logits, target):
        return jnp.take_along_axis(jax.nn.log_softmax(logits), target[..., None].astype(int), -1)[..., 0]
    def gaussian(y, mu, ls):
        return (-.5 * ((y - mu) * jnp.exp(-ls)) ** 2 - ls - .5 * jnp.log(2 * jnp.pi)).sum(-1)
    for h, o in outputs.items():
        if h == 'a':
            total += add(categorical(o['type_logits'], b['pitch_type']), v & b['type_valid'].astype(bool))
            idx = jnp.broadcast_to(b['pitch_type'][..., None, None], (*v.shape, 1, 5))
            mu = jnp.take_along_axis(o['stuff_mu'], idx, 2)[..., 0, :]
            ls = jnp.take_along_axis(o['stuff_logsigma'], idx, 2)[..., 0, :]
            total += add(gaussian(b['stuff'], mu, ls), tracked)
        elif h == 'b':
            for label, mask in [('swing', tracked), ('contact', tracked & (b['swing'] > 0)),
                                ('foul', tracked & (b['swing'] > 0) & (b['contact'] > 0)),
                                ('hbp', tracked & (b['swing'] == 0))]:
                total += add(-optax.sigmoid_binary_cross_entropy(o[label + '_logit'], b[label]), mask)
            if 'called_strike_logit' in o:
                # The call occurs only after a taken non-HBP pitch.  Keep this
                # separate from the old heads so pre-call-head checkpoints
                # retain their exact objective.
                call_mask = v & (b['swing'] == 0) & (b['hbp'] == 0)
                total += add(-optax.sigmoid_binary_cross_entropy(
                    o['called_strike_logit'], b['called_strike']), call_mask)
        elif h == 'c':
            if o['event_logits'].shape[-1] == 256:
                target = jnp.sum(b['events'].astype(jnp.int32) * (1 << jnp.arange(8)), -1)
                total += add(categorical(o['event_logits'], target), tracked & b.get('c_eligible', jnp.ones_like(v)))
            else:
                total += add(-optax.sigmoid_binary_cross_entropy(o['event_logits'], b['events']), tracked[..., None])
        else:
            measured = tracked & b['launch_valid'].astype(bool)
            total += add(gaussian(b['launch'], o['launch_mu'], o['launch_logsigma']), measured)
            total += add(categorical(o['outcome_logits'], b['batted_out']), measured & b['batted_valid'].astype(bool))
            if d_hr_weight:
                p_hr = jax.nn.softmax(o['outcome_logits'], axis=-1)[..., 4]
                y_hr = (b['batted_out'] == 4).astype(p_hr.dtype)
                hr_ll = y_hr * jnp.log(p_hr + 1e-6) + (1 - y_hr) * jnp.log(1 - p_hr + 1e-6)
                total += d_hr_weight * add(hr_ll, measured & b['batted_valid'].astype(bool))
    return total


def export_world(checkpoint, seed=0, sample=False):
    """Return all standalone heads from one coherent posterior world draw.

    Reuse the returned variables for every pitch/batch in that world. Evaluation
    years beyond the fitted range currently use the last fitted skill season.
    """
    c = checkpoint
    model = BayesianNetwork(c['options'], c['heads'], c['features'][0].shape[-1])
    skills = draw_skills(c['posterior'], jax.random.PRNGKey(seed), c['residual_scale'],
                        c['prior'], c['walk_scale'], sample)
    _, tables = model.apply({'params': c['network']}, c['example'], c['features'], skills, c['role_indices'])
    result = {}
    for h in c['heads']:
        params = deepcopy(c['network']['head_' + h])
        params['trunk']['super_state'] = deepcopy(c['network']['shared_ss'])
        result[h] = {'params': params, 'player_data': {'trunk': {'super_state': {
            role: tables[h][role][indices] for role, indices in c['role_indices'].items()}}}}
    return result


def pa_skill_features(pitches, registry, args):
    """Use PA's exact feature builder, remapped to ABCD's neutral-zero registry."""
    from diamondworldjax.scripts.train_pa import _build_player_table
    table = _build_player_table(pitches,
        recency_halflife=getattr(args, 'recency_halflife', None),
        contact_quality=getattr(args, 'contact_quality', False),
        per_stat_shrink=getattr(args, 'per_stat_shrink', False))
    stats = np.zeros((len(registry) + 1, table['stats'].shape[1]), np.float32)
    league, hand = np.zeros(len(registry) + 1, np.int32), np.zeros(len(registry) + 1, np.int32)
    for pid, target in registry.items():
        source = table['id_to_idx'].get(pid)
        if source is not None:
            stats[target], league[target], hand[target] = table['stats'][source], table['league'][source], table['hand'][source]
    return stats, league, hand


def pa_role_skill_features(pitches, registry, args):
    """Leakage-free deterministic features, separated by on-field role.

    ``_build_player_table`` intentionally summarizes batting outcomes.  Applying
    that same table to pitchers leaves their useful rate/count fields at zero.
    Re-running it with pitcher and batter IDs exchanged gives the matching
    opponent-outcome summary for a pitcher.  The learned encoder weights remain
    shared, while the evidence supplied to each role is now correct.
    """
    import polars as pl
    from diamondworldjax.scripts.train_pa import _build_player_table

    kw = dict(
        recency_halflife=getattr(args, "recency_halflife", None),
        contact_quality=getattr(args, "contact_quality", False),
        per_stat_shrink=getattr(args, "per_stat_shrink", False),
    )
    batter_table = _build_player_table(pitches, **kw)
    pitcher_col = "pitcher_id" if "pitcher_id" in pitches.columns else "pitcher_idx"
    batter_col = "batter_id" if "batter_id" in pitches.columns else "batter_idx"
    swapped = pitches.select(
        pl.col(pitcher_col).alias("batter_id"),
        pl.col(batter_col).alias("pitcher_id"),
        pl.all().exclude([pitcher_col, batter_col]),
    )
    pitcher_table = _build_player_table(swapped, **kw)

    feature_dim = batter_table["stats"].shape[-1]
    stats = np.zeros((2, len(registry) + 1, feature_dim), np.float32)
    league = np.zeros((2, len(registry) + 1), np.int32)
    hand = np.zeros((2, len(registry) + 1), np.int32)
    for pid, target in registry.items():
        batter_source = batter_table["id_to_idx"].get(pid)
        pitcher_source = pitcher_table["id_to_idx"].get(pid)
        if batter_source is not None:
            stats[1, target] = batter_table["stats"][batter_source]
            league[1, target] = batter_table["league"][batter_source]
            hand[1, target] = int(batter_table["bat_hand"][batter_source] >= .5)
        if pitcher_source is not None:
            stats[0, target] = pitcher_table["stats"][pitcher_source]
            league[0, target] = pitcher_table["league"][pitcher_source]
            # Handedness is metadata, not an outcome summary: retain the
            # original pitcher's throwing hand after the ID swap above.
            source = batter_table["id_to_idx"].get(pid)
            if source is not None:
                hand[0, target] = int(batter_table["pit_hand"][source] >= .5)

    # Column 4 is PA/BF exposure.  Its raw 0--2000+ scale overwhelms rate
    # features before LayerNorm; retain its confidence signal on a bounded,
    # monotone scale instead.
    stats[..., 4] = np.minimum(
        np.log1p(stats[..., 4]) / np.log1p(2500.0), 1.0
    )
    return stats, league, hand


def bayesian_marginal_prepare(model, params, features, skills, roles, *, train=False, key=None):
    from .marginal_pitch_likelihood import prepare_hidden
    def call(data, **kwargs):
        rngs = {'dropout': jax.random.fold_in(key, 892)} if train else None
        return model.apply({'params': params}, data, features, skills, roles,
                           train=train, rngs=rngs, **kwargs)[0]
    return prepare_hidden(lambda data: call(data, encode_only=True),
        lambda data, hidden, selected: call(data, hidden_override=hidden, selected_heads=selected))


def run_bayesian(args, train, test, maps, seasons, feature_pitches=None):
    from .pitchformer_checkpoint import save_metadata
    if not 0 < args.skill_residual_scale < 1 or args.skill_walk_scale <= 0:
        raise ValueError('Residual scale must be between 0 and 1; walk scale must be positive')
    ids = sorted(set(maps['pitcher']) | set(maps['batter']))
    registry = {pid: i + 1 for i, pid in enumerate(ids)}
    roles = {}
    for role in ('pitcher', 'batter'):
        indices = np.zeros(maps['n_' + role], np.int32)
        for pid, local in maps[role].items():
            indices[local] = registry[pid]
        roles[role] = jnp.asarray(indices)
    # Explicit feature input prevents accidental use of held-out season statistics.
    # Without it, the encoder has neutral covariates; skills are still learned.
    features = (np.zeros((len(ids) + 1, 1), np.float32),
                np.zeros(len(ids) + 1, np.int32), np.zeros(len(ids) + 1, np.int32))
    if args.skill_features:
        with np.load(args.skill_features, allow_pickle=False) as f:
            if int(f['through_year']) > max(seasons):
                raise ValueError('Skill features must not include held-out years')
            source = {int(pid): i for i, pid in enumerate(f['player_ids'])}
            role_aware = f['stats'].ndim == 3
            stats_shape = ((2, len(ids) + 1, f['stats'].shape[-1]) if role_aware
                           else (len(ids) + 1, f['stats'].shape[-1]))
            stats = np.zeros(stats_shape, np.float32)
            league = np.zeros(stats_shape[:-1], np.int32)
            hand = np.zeros(stats_shape[:-1], np.int32)
            for pid, target in registry.items():
                if pid in source:
                    i = source[pid]
                    if role_aware:
                        stats[:, target] = f['stats'][:, i]
                        league[:, target] = f['league'][:, i]
                        hand[:, target] = f['hand'][:, i]
                    else:
                        stats[target], league[target], hand[target] = f['stats'][i], f['league'][i], f['hand'][i]
            if not np.isfinite(stats).all() or not np.isin(league, [0, 1]).all() or not np.isin(hand, [0, 1]).all():
                raise ValueError('Invalid skill covariates')
            features = stats, league, hand
    elif getattr(args, 'skill_feature_mode', 'neutral') == 'pa':
        if feature_pitches is None:
            raise ValueError('PA-compatible features require training pitches')
        if not set(feature_pitches['season'].unique().to_list()).issubset(seasons):
            raise ValueError('Feature pitches include seasons outside training')
        features = pa_role_skill_features(feature_pitches, registry, args)
        print('Bayesian skills: role-aware training-only PA/BF covariates selected.', flush=True)
    else:
        print('Bayesian skills: neutral statistical covariates selected.', flush=True)
    features = tuple(jnp.asarray(x) for x in features)
    s = max(seasons) - min(seasons) + 1 if args.skill_prior == 'walk' else 1
    for arr in (train, test):
        arr['skill_season'] = np.clip(arr['season'] - min(seasons), 0, s - 1).astype(np.int32)
    options = dict(n_pitchers=maps['n_pitcher'], n_batters=maps['n_batter'], n_parks=maps['n_park'],
                   d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                   player_mode='pa', skill_seasons=s, pitch_history=args.pitch_history,
                   position_encoding=getattr(args, 'position_encoding', 'sinusoidal'),
                   window_size=getattr(args, 'window_size', 0),
                   observation_masks=True, c_event_mode=getattr(args, 'c_event_mode', 'legacy'),
                   c_support=getattr(args, 'c_support', None),
                   learned_called_strike=getattr(args, 'learned_called_strike', True))
    model = BayesianNetwork(options, args.stack, features[0].shape[-1], dropout=getattr(args, 'dropout', .1))
    example = {k: jnp.asarray(v[:1]) for k, v in train.items()}
    shape = (1 + len(args.stack), len(ids), s, SKILL_DIM)
    scales = np.asarray([np.sqrt(1 - args.skill_residual_scale ** 2)] +
                        [args.skill_residual_scale] * len(args.stack), np.float32)[:, None, None, None]
    posterior = dict(mu=jnp.zeros(shape), rho=jnp.broadcast_to(jnp.asarray(np.log(np.expm1(scales))), shape))
    key = jax.random.PRNGKey(args.seed)
    skills = draw_skills(posterior, key, args.skill_residual_scale, args.skill_prior, args.skill_walk_scale, False)
    network = model.init(key, example, features, skills, roles)['params']
    params = dict(network=network, posterior=posterior)
    schedule = optax.warmup_cosine_decay_schedule(init_value=args.lr * .1,
        peak_value=args.lr, warmup_steps=max(1, args.steps // 20),
        decay_steps=max(2, args.steps), end_value=args.lr * .05)
    # The posterior already has its explicit KL prior; decay network weights only.
    decay_mask = dict(network=jax.tree.map(lambda _: True, network),
                      posterior=jax.tree.map(lambda _: False, posterior))
    optimizer = optax.chain(optax.clip_by_global_norm(1.),
        optax.adamw(schedule, weight_decay=1e-4, mask=decay_mask))
    state = optimizer.init(params)
    n = len(train['valid'])
    def objective(p, batch, key):
        skills = draw_skills(p['posterior'], key, args.skill_residual_scale, args.skill_prior, args.skill_walk_scale)
        if getattr(args, 'missing_samples', 0):
            from .marginal_pitch_likelihood import marginal_log_likelihood
            apply = lambda data: model.apply({'params': p['network']}, data, features, skills, roles, train=True, rngs={'dropout': jax.random.fold_in(key, 892)})[0]
            ll = marginal_log_likelihood(apply, batch, jax.random.fold_in(key, 891), args.missing_samples,
                prepare=bayesian_marginal_prepare(model, p['network'], features, skills, roles,
                                                 train=True, key=key),
                d_hr_weight=getattr(args, 'd_hr_weight', 0.0))
        else:
            out, _ = model.apply({'params': p['network']}, batch, features, skills, roles, train=True, rngs={'dropout': jax.random.fold_in(key, 892)})
            ll = likelihood(out, batch, d_hr_weight=getattr(args, 'd_hr_weight', 0.0))
        # Uniform sequence sampling: sum likelihood * N/B; global KL exactly once.
        return (-ll * n / batch['valid'].shape[0] +
                skill_kl(p['posterior'], args.skill_residual_scale)) / n
    @jax.jit
    def update(p, state, batch, key):
        loss, grad = jax.value_and_grad(objective)(p, batch, key)
        updates, state = optimizer.update(grad, state, p)
        return optax.apply_updates(p, updates), state, loss
    from diamondworldjax.train.runtime import prefetch, tree_finite, bucket_batch
    chunk_size = getattr(args, 'update_chunk_size', 16)
    if chunk_size < 1:
        raise ValueError('update_chunk_size must be positive')
    @jax.jit
    def update_chunk(params, state, key, batches):
        def advance(carry, batch):
            p, opt_state, rng, failed = carry
            rng, subkey = jax.random.split(rng)
            proposed, next_state, loss = update(p, opt_state, batch, subkey)
            finite = jnp.isfinite(loss) & tree_finite((proposed, next_state))
            failed = failed | ~finite
            p, opt_state = jax.lax.cond(failed, lambda: (p, opt_state),
                                      lambda: (proposed, next_state))
            return (p, opt_state, rng, failed), loss
        return jax.lax.scan(advance, (params, state, key, jnp.array(False)), batches)

    rng = np.random.default_rng(args.seed)
    def training_chunks():
        step = 0
        while step < args.steps:
            count = min(chunk_size, args.steps-step, 100-((step-1) % 100))
            indices = rng.integers(n, size=(count, args.bs))
            yield jax.device_put(bucket_batch({k: v[indices] for k, v in train.items()}))
            step += count
    prepared = prefetch(training_chunks(), getattr(args, 'prefetch_depth', 2))
    step = 0
    try:
        for batches in prepared:
            (params, state, key, failed), values = update_chunk(params, state, key, batches)
            failed, values = jax.device_get((failed, values))
            if failed:
                raise FloatingPointError('Non-finite Bayesian objective or optimizer state')
            step += len(values)
            loss = values[-1]
            if (step-1) % 100 == 0:
                print(f'Bayesian ABCD step {step-1}: negative ELBO/sequence={float(loss):.4f}', flush=True)
    finally:
        prepared.close()
    checkpoint = dict(version=1, options=options, heads=args.stack, features=features,
        player_ids=ids, role_indices=roles, network=params['network'], posterior=params['posterior'],
        residual_scale=args.skill_residual_scale, prior=args.skill_prior, walk_scale=args.skill_walk_scale,
        example=example, optimizer_state=state, rng_key=key, steps=args.steps)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f'bayesian_{args.tag}.pkl', 'wb') as f:
        pickle.dump(jax.device_get(checkpoint), f)
    for h, variables in export_world(checkpoint).items():
        with open(outdir / f'{h.upper()}_{args.tag}_params.pkl', 'wb') as f:
            pickle.dump(jax.device_get(variables), f)
    save_metadata(args.out, args.tag, dict(version=1, config=vars(args), maps=maps,
        train_years=seasons, skill_season_base=min(seasons), bayesian=True,
        model_options=dict(dropout=getattr(args, 'dropout', .1), player_mode='pa', skill_seasons=s, pitch_history=args.pitch_history, residual_dim=0,
                           position_encoding=options['position_encoding'], window_size=options['window_size'],
                           observation_masks=True, c_event_mode=options['c_event_mode'], c_support=options['c_support'],
                           learned_called_strike=options['learned_called_strike'])))
    # Held-out likelihood at posterior means, under exactly the training masks.
    skills = draw_skills(params['posterior'], key, args.skill_residual_scale, args.skill_prior, args.skill_walk_scale, False)
    def heldout(batch):
        apply = lambda data: model.apply({'params': params['network']}, data, features, skills, roles)[0]
        if getattr(args, 'missing_samples', 0):
            from .marginal_pitch_likelihood import marginal_log_likelihood
            return marginal_log_likelihood(apply, batch, key, args.missing_samples,
                prepare=bayesian_marginal_prepare(model, params['network'], features, skills, roles))
        return likelihood(apply(batch), batch)
    score = jax.jit(heldout)
    ll = sum(float(score({k: jnp.asarray(v[i:i + args.bs]) for k, v in test.items()}))
             for i in range(0, len(test['valid']), args.bs))
    valid = test['valid'] * test.get('loss_mask', 1) > 0
    tracked = valid & (test['type_valid'] > 0) & (test['stuff_valid'] > 0)
    measured = tracked & (test['launch_valid'] > 0)
    coverage = dict(valid_pitches=int(valid.sum()), type=int((valid & (test['type_valid'] > 0)).sum()),
        tracked=int(tracked.sum()), contact=int((tracked & (test['swing'] > 0)).sum()),
        foul=int((tracked & (test['swing'] > 0) & (test['contact'] > 0)).sum()),
        hbp=int((tracked & (test['swing'] == 0)).sum()), launch=int(measured.sum()),
        outcome=int((measured & (test['batted_valid'] > 0)).sum()))
    if getattr(args, 'missing_samples', 0):
        coverage = dict(valid_pitches=int(valid.sum()),
            type=int((valid & (test['type_valid'] > 0)).sum()),
            stuff_components=int((valid[..., None] & test.get('stuff_observed', np.broadcast_to(test['stuff_valid'][..., None] > 0, test['stuff'].shape))).sum()),
            launch_components=int((valid[..., None] & test.get('launch_observed', np.broadcast_to(test['launch_valid'][..., None] > 0, test['launch'].shape))).sum()),
            outcome=int((valid & (test['batted_valid'] > 0)).sum()),
            c_eligible=int((valid & test.get('c_eligible', np.ones_like(valid))).sum()))
    report = dict(d_interpretation="Associative launch/outcome factorization; park/environment enter both stages and outcome can bypass launch. Ablations are predictive, not causal.", history_missingness='mask-aware encoder; no latent-history integration', config=vars(args), heads=args.stack, heldout_log_likelihood=ll,
        heldout_target_counts=coverage, prediction_mode='posterior_mean_skills',
        final_negative_elbo_per_sequence=float(loss),
        statistical_covariates=('provided_role_aware' if args.skill_features and features[0].ndim == 3
                                else 'provided' if args.skill_features
                                else 'pa_role_aware' if getattr(args, 'skill_feature_mode', 'neutral') == 'pa'
                                else 'neutral'))
    with open(outdir / f'bayesian_{args.tag}_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Saved Bayesian posterior and mean head exports. Held-out summed log likelihood: {ll:.4f}', flush=True)
    return checkpoint
