"""World-model technique sweep, measured by what actually matters.

Per-PA NLL is saturated: the marginal outcome entropy is ~1.495 nats and every
conditional model scores near or above it, because one PA is dominated by
irreducible noise. The real measure of a PA-level world model is whether its
CONDITIONAL distributions are right, i.e. does it rank players correctly and stay
calibrated. So this harness sweeps architectures/regularizers and scores each by
cross-player rate correlation (K%, BB%, hit%, HR%) plus NLL and marginal-L1.

Identical inputs (batter/pitcher/park embeddings + rate stats + state [+platoon]),
identical discriminative training with val early-stopping; only the config differs.
Train 2015-2022 (10% val), test 2023-2024.

  --arch {mlp,transformer,gru,lstm}   --layers N --dm D --heads H
  --dropout p --embed-dropout p --wd w --label-smooth s --ensemble K --platoon
"""
from __future__ import annotations
import argparse, time
import numpy as np, polars as pl
from diamondworldjax.scripts.seq_models import build_seqs, pitcher_rates, TRAIN, TEST, HITS
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root

NOUT = 9
K, BBs, HRs = 2, (1,), 6  # class indices: K=2, BB=1(+HBP=2? no) ... use masks below
KIDX, BBIDX, HBP, HRIDX = 2, 1, 2, 6
HIT_IDX = [3, 4, 5, 6]  # 1B,2B,3B,HR
BB_IDX = [1, 2]         # BB,HBP


def build_statcast(pitches, id2i, P):
    """Leakage-free per-player Statcast descriptors from ALL training pitches.
    Batter (5): swing-rate, whiff-rate, mean exit velo, mean launch angle, hard-hit
    rate. Pitcher (3): mean velocity, mean movement, induced whiff-rate. These
    quality-of-contact / stuff metrics predict true talent better than outcome
    rates (xStats thesis). Unknowns filled with the league mean."""
    bcol = "batter_id" if "batter_id" in pitches.columns else "batter_idx"
    pcol = "pitcher_id" if "pitcher_id" in pitches.columns else "pitcher_idx"
    df = pitches.with_columns((pl.col("swing") & ~pl.col("contact")).alias("_whiff"),
                              (pl.col("pfx_x") ** 2 + pl.col("pfx_z") ** 2).sqrt().alias("_mov"))
    bb = df.filter(pl.col("launch_speed").is_not_null())
    bat_sc = np.full((P, 5), np.nan, np.float32)
    g = df.group_by(bcol).agg([pl.col("swing").mean().alias("sw"),
        pl.col("_whiff").sum().alias("wh"), pl.col("swing").sum().alias("nsw"), pl.len().alias("n")])
    gb = bb.group_by(bcol).agg([pl.col("launch_speed").mean().alias("ev"),
        pl.col("launch_angle").mean().alias("la"), (pl.col("launch_speed") >= 95).mean().alias("hh"),
        pl.len().alias("nb")]).join(g, on=bcol, how="left")
    for r in gb.iter_rows(named=True):
        i = id2i.get(int(r[bcol]))
        if i is not None and (r["n"] or 0) >= 100:
            bat_sc[i] = [r["sw"], (r["wh"] or 0) / max(r["nsw"] or 1, 1),
                         r["ev"], r["la"], r["hh"]]
    pit_sc = np.full((P, 3), np.nan, np.float32)
    gp = df.group_by(pcol).agg([pl.col("release_speed").mean().alias("velo"),
        pl.col("_mov").mean().alias("mov"), pl.col("_whiff").sum().alias("wh"),
        pl.col("swing").sum().alias("nsw"), pl.len().alias("n")])
    for r in gp.iter_rows(named=True):
        i = id2i.get(int(r[pcol]))
        if i is not None and (r["n"] or 0) >= 100:
            pit_sc[i] = [r["velo"], r["mov"], (r["wh"] or 0) / max(r["nsw"] or 1, 1)]
    bat_sc[np.isnan(bat_sc).any(1)] = np.nanmean(bat_sc, 0)
    pit_sc[np.isnan(pit_sc).any(1)] = np.nanmean(pit_sc, 0)
    return bat_sc, pit_sc


def make_factory(nplayers, arch, layers, dm, heads, dropout, embed_dropout, no_player_emb=False):
    import jax.numpy as jnp, flax.linen as nn
    class Model(nn.Module):
        @nn.compact
        def __call__(self, bat, pit, park, br, pr, st, plat=None, train=False):
            pk = nn.Embed(100, 8)
            feats = [pk(park), br, pr, st]
            if not no_player_emb:
                pe = nn.Embed(nplayers, 32)
                b, p = pe(bat), pe(pit)
                if embed_dropout > 0:
                    b = nn.Dropout(embed_dropout, deterministic=not train)(b)
                    p = nn.Dropout(embed_dropout, deterministic=not train)(p)
                feats = [b, p] + feats
            if plat is not None:
                feats.append(plat)
            x = nn.Dense(dm)(jnp.concatenate(feats, -1))
            T = bat.shape[1]
            if arch == "mlp":
                for _ in range(layers):
                    x = nn.relu(nn.Dense(dm)(x))
                    if dropout > 0: x = nn.Dropout(dropout, deterministic=not train)(x)
            elif arch == "transformer":
                pos = self.param("pos", nn.initializers.normal(0.02), (1, 90, dm))
                x = x + pos[:, :T]
                cm = jnp.tril(jnp.ones((T, T), bool))[None, None]
                for _ in range(layers):
                    h = nn.LayerNorm()(x)
                    h = nn.MultiHeadDotProductAttention(num_heads=heads, qkv_features=dm,
                        dropout_rate=dropout, deterministic=not train)(h, h, mask=cm)
                    x = x + h
                    h = nn.LayerNorm()(x)
                    h = nn.Dense(dm)(nn.gelu(nn.Dense(4 * dm)(h)))
                    if dropout > 0: h = nn.Dropout(dropout, deterministic=not train)(h)
                    x = x + h
                x = nn.LayerNorm()(x)
            elif arch in ("gru", "lstm"):
                cell = nn.GRUCell(dm) if arch == "gru" else nn.OptimizedLSTMCell(dm)
                for _ in range(layers):
                    x = nn.RNN(cell)(x)
                    if dropout > 0: x = nn.Dropout(dropout, deterministic=not train)(x)
            return nn.Dense(NOUT)(x)
    return Model()


def run(cfg):
    import jax, jax.numpy as jnp, optax
    t0 = time.time()
    train_seasons = list(range(2015, cfg.get("train_end", 2022) + 1))
    test_seasons = ([int(x) for x in cfg["test_seasons"].split(",")]
                    if cfg.get("test_seasons") else TEST)
    trp = load_seasons(train_seasons, data_root=processed_root())
    ptab = _build_player_table(trp, recency_halflife=cfg.get("recency_halflife"))
    park_map = _build_park_index(trp)
    pit = pitcher_rates(trp.filter(pl.col("pa_terminal")), ptab["id_to_idx"], len(ptab["hand"]))
    bat_sc = pit_sc = None
    if cfg.get("statcast"):
        bat_sc, pit_sc = build_statcast(trp, ptab["id_to_idx"], len(ptab["hand"]))
    del trp
    if cfg.get("mle"):
        # append rookies (unseen in training) with translated minor-league rate features;
        # their fresh embedding stays near init (no train PAs) so the prediction rides on
        # the MLE rate features, de-blanking them instead of the shared unknown slot.
        mle = np.load(cfg["mle"]); rids = mle["ids"].astype(int); rrates = mle["rates"]
        id2i = ptab["id_to_idx"]; new = [(r, rr) for r, rr in zip(rids, rrates) if int(r) not in id2i]
        if new:
            P0, F = ptab["stats"].shape
            add = np.zeros((len(new), F), np.float32)
            for j, (_, rr) in enumerate(new):
                add[j, 0], add[j, 1], add[j, 2], add[j, 3] = rr; add[j, 4] = 150.0
            ptab["stats"] = np.concatenate([ptab["stats"], add], 0)
            for hk in ("hand",):
                ptab[hk] = np.concatenate([ptab[hk], np.zeros(len(new), ptab[hk].dtype)])
            for j, (r, _) in enumerate(new):
                id2i[int(r)] = P0 + j
    tr = build_seqs(train_seasons, ptab, pit, park_map)
    te = build_seqs(test_seasons, ptab, pit, park_map)
    if bat_sc is not None:
        for d in (tr, te):
            d["sc"] = np.concatenate([bat_sc[d["bat"]], pit_sc[d["pit"]]], -1).astype(np.float32)
    P = len(ptab["hand"])
    G = tr["y"].shape[0]
    rng_np = np.random.default_rng(0)
    perm = rng_np.permutation(G); nval = int(G * 0.1)
    val_i, fit_i = perm[:nval], perm[nval:]

    # standardize the continuous features (rate stats + state) on the fit set so the
    # small-magnitude rate stats are not swamped by the ~unit-scale player embeddings.
    vfit = tr["valid"][fit_i].reshape(-1)
    std_keys = ["bat_rate", "pit_rate", "state"] + (["sc"] if "sc" in tr else [])
    for key in std_keys:
        flat = tr[key][fit_i].reshape(-1, tr[key].shape[-1])[vfit]
        mu, sd = flat.mean(0), flat.std(0)
        # floor near-constant features (e.g. shift/clock are constant 0 in 2015-22
        # but 1 in 2023-24; sd~0 would blow up the test values). sd=1 keeps them raw.
        sd = np.where(sd < 1e-2, 1.0, sd)
        tr[key] = ((tr[key] - mu) / sd).astype(np.float32)
        te[key] = ((te[key] - mu) / sd).astype(np.float32)

    def feats(d, idx, plat=False):
        out = [jnp.array(d["bat"][idx]), jnp.array(d["pit"][idx]), jnp.array(d["park"][idx]),
               jnp.array(d["bat_rate"][idx]), jnp.array(d["pit_rate"][idx]), jnp.array(d["state"][idx])]
        if "sc" in d:
            out.append(jnp.array(d["sc"][idx]))   # Statcast features via the model's extra (plat) slot
        return out

    def build_one(seed):
        import jax
        model = make_factory(P, cfg["arch"], cfg["layers"], cfg["dm"], cfg["heads"],
                             cfg["dropout"], cfg["embed_dropout"], cfg["no_player_emb"])
        rng = jax.random.PRNGKey(seed)
        params = model.init({"params": rng, "dropout": rng}, *feats(tr, fit_i[:4]), train=False)
        sched = optax.cosine_decay_schedule(cfg["lr"], cfg["steps"]) if cfg["cosine"] else cfg["lr"]
        opt = optax.adamw(sched, weight_decay=cfg["wd"]); st = opt.init(params)
        ls = cfg["label_smooth"]

        def loss_fn(params, f, y, v, drng):
            logits = model.apply(params, *f, train=True, rngs={"dropout": drng})
            yc = jnp.clip(y, 0, NOUT - 1)
            if ls > 0:
                oh = jax.nn.one_hot(yc, NOUT) * (1 - ls) + ls / NOUT
                ll = optax.softmax_cross_entropy(logits, oh)
            else:
                ll = optax.softmax_cross_entropy_with_integer_labels(logits, yc)
            return (ll * v).sum() / v.sum()

        @jax.jit
        def step(params, st, f, y, v, drng):
            l, g = jax.value_and_grad(loss_fn)(params, f, y, v, drng)
            upd, st = opt.update(g, st, params); return optax.apply_updates(params, upd), st, l

        @jax.jit
        def logits_of(params, f): return model.apply(params, *f, train=False)

        def val_nll(params):
            s = n = 0.0
            for i in range(0, len(val_i), 256):
                idx = val_i[i:i+256]
                lg = np.array(logits_of(params, feats(tr, idx)))
                y = tr["y"][idx]; v = tr["valid"][idx]
                pp = np.exp(lg - lg.max(-1, keepdims=True)); pp /= pp.sum(-1, keepdims=True)
                m = v & (y >= 0); yi = np.clip(y, 0, NOUT - 1)
                s += -np.log(np.clip(pp[m, yi[m]], 1e-7, 1)).sum(); n += m.sum()
            return s / n

        drng = jax.random.PRNGKey(seed + 1)
        best, bestp, wait = 1e9, params, 0
        for i in range(cfg["steps"]):
            idx = rng_np.integers(0, len(fit_i), cfg["batch"]); idx = fit_i[idx]
            drng, k = jax.random.split(drng)
            params, st, l = step(params, st, feats(tr, idx),
                                  jnp.array(tr["y"][idx]), jnp.array(tr["valid"][idx], jnp.float32), k)
            if (i + 1) % 400 == 0:
                v = val_nll(params)
                if v < best - 1e-4: best, bestp, wait = v, params, 0
                else:
                    wait += 400
                    if wait >= cfg["patience"]: break
        return bestp, logits_of

    # ensemble average of probabilities
    probs_sum = None
    for s in range(cfg["ensemble"]):
        bestp, logits_of = build_one(cfg["seed"] + s)
        Gte = te["y"].shape[0]
        pr = np.zeros((Gte, te["y"].shape[1], NOUT), np.float32)
        for i in range(0, Gte, 128):
            idx = np.arange(i, min(i + 128, Gte))
            lg = np.array(logits_of(bestp, feats(te, idx)))
            e = np.exp(lg - lg.max(-1, keepdims=True)); pr[idx] = e / e.sum(-1, keepdims=True)
        probs_sum = pr if probs_sum is None else probs_sum + pr
    P_te = probs_sum / cfg["ensemble"]

    # metrics
    y = te["y"]; v = te["valid"]; m = v & (y >= 0); yi = np.clip(y, 0, NOUT - 1)
    pm = P_te[m]; ym = yi[m]
    nll = -np.log(np.clip(pm[np.arange(len(ym)), ym], 1e-7, 1)).mean()
    acc = (pm.argmax(-1) == ym).mean()
    marg = pm.mean(0); realm = np.bincount(ym, minlength=NOUT) / len(ym)
    margL1 = np.abs(marg - realm).sum()

    # player-stat reproduction: per-batter predicted vs real rates
    bat = te["bat"][m]
    predK = pm[:, KIDX]; predBB = pm[:, BB_IDX].sum(1); predHit = pm[:, HIT_IDX].sum(1); predHR = pm[:, HRIDX]
    realK = (ym == KIDX); realBB = np.isin(ym, BB_IDX); realHit = np.isin(ym, HIT_IDX); realHR = (ym == HRIDX)
    def corr_by_player(pred, real, minpa=150):
        order = np.argsort(bat)
        b = bat[order]; pv = pred[order]; rv = real[order].astype(float)
        uniq, start = np.unique(b, return_index=True)
        ps, rs, ns = [], [], []
        for j in range(len(uniq)):
            a = start[j]; z = start[j+1] if j+1 < len(uniq) else len(b)
            if z - a >= minpa:
                ps.append(pv[a:z].mean()); rs.append(rv[a:z].mean()); ns.append(z-a)
        ps, rs = np.array(ps), np.array(rs)
        return float(np.corrcoef(ps, rs)[0, 1]), len(ps)
    if cfg.get("save_rates"):
        sK = np.zeros(P); sBB = np.zeros(P); sHit = np.zeros(P); sHR = np.zeros(P)
        rrK = np.zeros(P); rrBB = np.zeros(P); rrHit = np.zeros(P); rrHR = np.zeros(P); cc = np.zeros(P)
        np.add.at(sK, bat, predK); np.add.at(sBB, bat, predBB); np.add.at(sHit, bat, predHit); np.add.at(sHR, bat, predHR)
        np.add.at(rrK, bat, realK.astype(float)); np.add.at(rrBB, bat, realBB.astype(float))
        np.add.at(rrHit, bat, realHit.astype(float)); np.add.at(rrHR, bat, realHR.astype(float)); np.add.at(cc, bat, 1.0)
        np.savez("data/eval2/mlp_rates.npz", sumK=sK, sumBB=sBB, sumHit=sHit, sumHR=sHR,
                 rK=rrK, rBB=rrBB, rHit=rrHit, rHR=rrHR, cnt=cc)
    cK, nplayers = corr_by_player(predK, realK)
    cBB, _ = corr_by_player(predBB, realBB)
    cHit, _ = corr_by_player(predHit, realHit)
    cHR, _ = corr_by_player(predHR, realHR)
    avg_corr = np.mean([cK, cBB, cHit, cHR])

    tag = f"{cfg['arch']}_L{cfg['layers']}_d{cfg['dm']}_dr{cfg['dropout']}_ed{cfg['embed_dropout']}_wd{cfg['wd']}_ls{cfg['label_smooth']}_ens{cfg['ensemble']}" + ("_noPE" if cfg.get("no_player_emb") else "") + ("_SC" if cfg.get("statcast") else "")
    line = (f"{tag:52s} NLL {nll:.4f} acc {acc:.4f} margL1 {margL1:.4f} | "
            f"corr K {cK:.3f} BB {cBB:.3f} Hit {cHit:.3f} HR {cHR:.3f} AVG {avg_corr:.3f} "
            f"(np={nplayers}, {time.time()-t0:.0f}s)")
    print(line, flush=True)
    with open("data/eval2/wm_sweep.txt", "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="transformer")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dm", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--embed-dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--platoon", action="store_true")
    ap.add_argument("--no-player-emb", action="store_true")
    ap.add_argument("--statcast", action="store_true", help="Add per-player Statcast stuff/contact features.")
    ap.add_argument("--save-rates", action="store_true")
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--patience", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-end", type=int, default=2022,
                    help="Last training season (inclusive). 2023 folds in the previous "
                         "season; then test 2024 only. Matches v15 for cross-arch comparison.")
    ap.add_argument("--test-seasons", default="2023,2024",
                    help="Comma list of eval seasons; use 2024 when train-end is 2023.")
    ap.add_argument("--recency-halflife", type=float, default=None,
                    help="Recency-weight the player rate features (seasons); pass 2.0 to match "
                         "v15 so the previous season is weighted highest.")
    ap.add_argument("--mle", default=None,
                    help="Path to mle_rates.npz; inject translated minor-league rate features "
                         "for rookies unseen in training.")
    args = ap.parse_args()
    run(vars(args))


if __name__ == "__main__":
    main()
