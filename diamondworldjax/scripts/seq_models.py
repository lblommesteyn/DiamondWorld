"""Transformer and JEPA experiments for per-PA outcome prediction.

The production model predicts each plate appearance independently from a
hand-crafted context vector (an MLP head). This asks whether sequence structure
(a causal transformer over the game's PA stream: times-through-order, fatigue
trajectory, momentum) or learned latent-predictive pretraining (JEPA) extract
signal the context-MLP misses. To isolate ARCHITECTURE from training regime,
all models share identical inputs and the same discriminative training loop; only
the network differs.

Inputs per PA token: learned batter/pitcher/park embeddings + batter rate stats +
pitcher allowed rates + 8 game-state scalars. Target: the 9-way PA outcome.
Train 2015-2022, test 2023-2024. Metrics: per-PA test NLL (vs the production
model's ~1.52), accuracy, and marginal-distribution L1.

  --arch mlp          per-PA MLP baseline (no sequence)
  --arch transformer  causal transformer over the game PA sequence
  --arch jepa         JEPA pretrain (predict masked-PA latent, VICReg) + linear probe

Usage: python -m diamondworldjax.scripts.seq_models --arch transformer --steps 4000
"""
from __future__ import annotations
import argparse, time
from functools import partial
import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.data.pa_batching import build_pa_batch
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index, apply_park_idx

TRAIN = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST = [2023, 2024]
HITS = ("1B", "2B", "3B", "HR")
NOUT = 9
MAXT = 90


def pitcher_rates(train_pa, id2i, P):
    pcol = "pitcher_id" if "pitcher_id" in train_pa.columns else "pitcher_idx"
    g = (train_pa.filter(pl.col("pa_outcome").is_not_null()).group_by(pcol).agg([
        pl.col("pa_outcome").is_in(HITS).mean().alias("hit"),
        pl.col("pa_outcome").is_in(["BB", "HBP"]).mean().alias("bb"),
        (pl.col("pa_outcome") == "K").mean().alias("k"),
        (pl.col("pa_outcome") == "HR").mean().alias("hr"), pl.len().alias("n")]))
    out = np.full((P, 4), np.nan, np.float32)
    for r in g.iter_rows(named=True):
        i = id2i.get(int(r[pcol]))
        if i is not None and r["n"] >= 50:
            out[i] = [r["hit"], r["bb"], r["k"], r["hr"]]
    out[np.isnan(out).any(1)] = np.nanmean(out, 0)
    return out


def build_seqs(seasons, ptab, pit, park_map):
    """Return per-game padded arrays via build_pa_batch, concatenated."""
    id2i = ptab["id_to_idx"]
    df = load_seasons(seasons, data_root=processed_root()).filter(pl.col("pa_terminal"))
    df = apply_park_idx(df, park_map)
    gids = df["game_pk"].unique().to_numpy()
    chunks = []
    for i in range(0, len(gids), 512):
        sub = df.filter(pl.col("game_pk").is_in(gids[i:i+512].tolist()))
        b = build_pa_batch(sub)
        bat_ids = np.vectorize(lambda x: id2i.get(int(x), 0))(np.array(b["batter_ids"]))
        pit_ids = np.vectorize(lambda x: id2i.get(int(x), 0))(np.array(b["pitcher_ids"]))
        state = np.stack([np.array(b[k]) for k in
                          ("inning", "half", "outs", "base_state", "score_diff", "tto",
                           "shift_restricted", "pitch_clock")], -1).astype(np.float32)
        chunks.append(dict(
            bat=bat_ids.astype(np.int32), pit=pit_ids.astype(np.int32),
            park=np.array(b["park_ids"]).astype(np.int32), state=state,
            y=np.array(b["pa_outcome"]).astype(np.int32), valid=np.array(b["pa_valid"])))
    out = {k: np.concatenate([c[k] for c in chunks], 0) for k in chunks[0]}
    out["bat_rate"] = ptab["stats"][:, :4][out["bat"]]
    out["pit_rate"] = pit[out["pit"]]
    return out


# ---------------- models (Flax) ----------------
def make_models():
    import jax, jax.numpy as jnp, flax.linen as nn

    P_EMB, PARK_EMB, DM = 32, 8, 128

    class Tokenizer(nn.Module):
        n_players: int
        n_parks: int = 100
        @nn.compact
        def __call__(self, bat, pit, park, bat_rate, pit_rate, state):
            pe = nn.Embed(self.n_players, P_EMB)
            pk = nn.Embed(self.n_parks, PARK_EMB)
            tok = jnp.concatenate([pe(bat), pe(pit), pk(park), bat_rate, pit_rate, state], -1)
            return nn.Dense(DM)(tok)

    class SeqMLP(nn.Module):
        n_players: int
        @nn.compact
        def __call__(self, b, p, k, br, pr, st, train=False):
            x = Tokenizer(self.n_players)(b, p, k, br, pr, st)
            x = nn.relu(nn.Dense(DM)(nn.relu(x)))
            x = nn.relu(nn.Dense(DM)(x))
            return nn.Dense(NOUT)(x)

    class Block(nn.Module):
        @nn.compact
        def __call__(self, x, mask):
            h = nn.LayerNorm()(x)
            h = nn.MultiHeadDotProductAttention(num_heads=4, qkv_features=DM)(h, h, mask=mask)
            x = x + h
            h = nn.LayerNorm()(x)
            h = nn.Dense(DM)(nn.gelu(nn.Dense(4 * DM)(h)))
            return x + h

    class SeqTransformer(nn.Module):
        n_players: int
        n_layers: int = 3
        @nn.compact
        def __call__(self, b, p, k, br, pr, st, train=False):
            T = b.shape[1]
            x = Tokenizer(self.n_players)(b, p, k, br, pr, st)
            pos = self.param("pos", nn.initializers.normal(0.02), (1, MAXT, DM))
            x = x + pos[:, :T]
            causal = jnp.tril(jnp.ones((T, T), bool))[None, None]  # (1,1,T,T)
            for _ in range(self.n_layers):
                x = Block()(x, causal)
            return nn.Dense(NOUT)(nn.LayerNorm()(x))

    return jax, jnp, nn, SeqMLP, SeqTransformer, Tokenizer, DM


# Authoritative class encoding (rules_engine.PA_OUTCOMES): 0=K 1=BB 2=HBP 3=1B 4=2B 5=3B 6=HR 7=out 8=E
KIDX, HRIDX, HIT_IDX, BB_IDX = 0, 6, [3, 4, 5, 6], [1, 2]


def player_corr(bat, probs, y, tag, out_path):
    """Cross-player rate correlation (predicted vs real K/BB/hit/HR), the world-model
    metric: does the model rank hitters correctly. Batters with >=150 test PAs."""
    m = (y >= 0)
    bat, probs, yi = bat[m], probs[m], np.clip(y[m], 0, NOUT - 1)
    P = int(bat.max()) + 1
    acc = {k: np.zeros(P) for k in ("pK", "pBB", "pHit", "pHR", "rK", "rBB", "rHit", "rHR", "n")}
    np.add.at(acc["pK"], bat, probs[:, KIDX]); np.add.at(acc["pHR"], bat, probs[:, HRIDX])
    np.add.at(acc["pBB"], bat, probs[:, BB_IDX].sum(1)); np.add.at(acc["pHit"], bat, probs[:, HIT_IDX].sum(1))
    np.add.at(acc["rK"], bat, (yi == KIDX)); np.add.at(acc["rHR"], bat, (yi == HRIDX))
    np.add.at(acc["rBB"], bat, np.isin(yi, BB_IDX)); np.add.at(acc["rHit"], bat, np.isin(yi, HIT_IDX))
    np.add.at(acc["n"], bat, 1.0)
    keep = acc["n"] >= 150
    def c(a, b): return float(np.corrcoef(acc[a][keep] / acc["n"][keep], acc[b][keep] / acc["n"][keep])[0, 1])
    cK, cBB, cHit, cHR = c("pK", "rK"), c("pBB", "rBB"), c("pHit", "rHit"), c("pHR", "rHR")
    line = (f"{tag} | corr K {cK:.3f} BB {cBB:.3f} Hit {cHit:.3f} HR {cHR:.3f} "
            f"AVG {np.mean([cK,cBB,cHit,cHR]):.3f} (np={int(keep.sum())})")
    print(line, flush=True); open(out_path, "a").write(line + "\n")
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "transformer", "jepa"], default="transformer")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-end", type=int, default=2022,
                    help="Last training season (2023 folds in the previous season; test 2024).")
    ap.add_argument("--test-seasons", default="2023,2024")
    ap.add_argument("--recency-halflife", type=float, default=None,
                    help="Recency-weight rate features (pass 2.0 to match v15).")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    train_seasons = list(range(2015, args.train_end + 1))
    test_seasons = [int(x) for x in args.test_seasons.split(",")]

    jax, jnp, nn, SeqMLP, SeqTransformer, Tokenizer, DM = make_models()
    import optax

    print("Building player table + sequences...", flush=True)
    train_pitches = load_seasons(train_seasons, data_root=processed_root())
    ptab = _build_player_table(train_pitches, recency_halflife=args.recency_halflife)
    park_map = _build_park_index(train_pitches)
    pit = pitcher_rates(train_pitches.filter(pl.col("pa_terminal")), ptab["id_to_idx"], len(ptab["hand"]))
    del train_pitches
    tr = build_seqs(train_seasons, ptab, pit, park_map)
    te = build_seqs(test_seasons, ptab, pit, park_map)
    P = len(ptab["hand"])
    print(f"  train games {tr['y'].shape[0]}  test games {te['y'].shape[0]}  players {P}", flush=True)

    def batch_feats(d, idx):
        return (jnp.array(d["bat"][idx]), jnp.array(d["pit"][idx]), jnp.array(d["park"][idx]),
                jnp.array(d["bat_rate"][idx]), jnp.array(d["pit_rate"][idx]), jnp.array(d["state"][idx]))

    if args.arch in ("mlp", "transformer"):
        Model = SeqMLP if args.arch == "mlp" else SeqTransformer
        model = Model(n_players=P)
        rng = jax.random.PRNGKey(args.seed)
        idx0 = np.arange(min(4, tr["y"].shape[0]))
        params = model.init(rng, *batch_feats(tr, idx0))
        opt = optax.adamw(args.lr, weight_decay=1e-4); st = opt.init(params)

        def loss_fn(params, feats, y, valid):
            logits = model.apply(params, *feats)
            ll = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.clip(y, 0, NOUT - 1))
            return (ll * valid).sum() / valid.sum()

        @jax.jit
        def step(params, st, feats, y, valid):
            l, g = jax.value_and_grad(loss_fn)(params, feats, y, valid)
            upd, st = opt.update(g, st, params); return optax.apply_updates(params, upd), st, l

        rng2 = np.random.default_rng(args.seed)
        G = tr["y"].shape[0]; t0 = time.time()
        for i in range(args.steps):
            idx = rng2.integers(0, G, args.batch)
            params, st, l = step(params, st, batch_feats(tr, idx),
                                  jnp.array(tr["y"][idx]), jnp.array(tr["valid"][idx], jnp.float32))
            if i % 500 == 0:
                print(f"  step {i}  train-nll {float(l):.4f}  {time.time()-t0:.0f}s", flush=True)

        # ---- eval on test ----
        @jax.jit
        def eval_logits(params, feats): return model.apply(params, *feats)
        nll_sum = n = correct = 0.0
        marg = np.zeros(NOUT); realm = np.zeros(NOUT)
        Gte = te["y"].shape[0]
        for i in range(0, Gte, 128):
            idx = np.arange(i, min(i + 128, Gte))
            logits = np.array(eval_logits(params, batch_feats(te, idx)))
            y = te["y"][idx]; v = te["valid"][idx]
            p = np.exp(logits - logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
            m = v & (y >= 0)
            yi = np.clip(y, 0, NOUT - 1)
            nll_sum += -np.log(np.clip(p[m, yi[m]], 1e-7, 1)).sum()
            correct += (p[m].argmax(-1) == yi[m]).sum(); n += m.sum()
            for j in range(NOUT):
                marg[j] += p[m][:, j].sum(); realm[j] += (yi[m] == j).sum()
        marg /= marg.sum(); realm /= realm.sum()
        print(f"\n=== {args.arch.upper()} TEST (2023-24, {int(n)} PAs) ===", flush=True)
        print(f"  per-PA NLL {nll_sum/n:.4f}   accuracy {correct/n:.4f}   marg-L1 {np.abs(marg-realm).sum():.4f}", flush=True)
        print("  (production SVI model per-PA NLL ~1.52 for reference)", flush=True)
        open("data/eval2/seq_%s.txt" % args.arch, "w").write(
            f"{args.arch} test NLL {nll_sum/n:.4f} acc {correct/n:.4f} marg-L1 {np.abs(marg-realm).sum():.4f} n {int(n)}\n")

    else:
        run_jepa(args, jax, jnp, nn, Tokenizer, DM, tr, te, P, optax)


def run_jepa(args, jax, jnp, nn, Tokenizer, DM, tr, te, P, optax):
    """JEPA: mask a PA, encode context causally, predict the masked PA's target
    embedding in latent space (VICReg anti-collapse). Then linear-probe the frozen
    context representation for outcome prediction, vs an outcome-embedding target."""
    import numpy as np, time

    class Ctx(nn.Module):
        n_players: int
        @nn.compact
        def __call__(self, b, p, k, br, pr, st):
            x = Tokenizer(self.n_players)(b, p, k, br, pr, st)
            pos = self.param("pos", nn.initializers.normal(0.02), (1, 90, DM))
            x = x + pos[:, :b.shape[1]]
            causal = jnp.tril(jnp.ones((b.shape[1],) * 2, bool))[None, None]
            for _ in range(3):
                h = nn.LayerNorm()(x)
                h = nn.MultiHeadDotProductAttention(num_heads=4, qkv_features=DM)(h, h, mask=causal)
                x = x + h
                h = nn.LayerNorm()(x)
                x = x + nn.Dense(DM)(nn.gelu(nn.Dense(4 * DM)(h)))
            return nn.LayerNorm()(x)          # per-position context repr (causal => excludes current outcome)

    class Predictor(nn.Module):
        @nn.compact
        def __call__(self, z):
            return nn.Dense(DM)(nn.gelu(nn.Dense(DM)(z)))

    # target = a learned embedding of the actual outcome class (the "world" token)
    ctx = Ctx(n_players=P); pred = Predictor()
    tgt_emb = nn.Embed(NOUT, DM)
    rng = jax.random.PRNGKey(args.seed)
    idx0 = np.arange(4)
    f0 = (jnp.array(tr["bat"][idx0]), jnp.array(tr["pit"][idx0]), jnp.array(tr["park"][idx0]),
          jnp.array(tr["bat_rate"][idx0]), jnp.array(tr["pit_rate"][idx0]), jnp.array(tr["state"][idx0]))
    pc = ctx.init(rng, *f0); pp = pred.init(rng, jnp.zeros((4, 90, DM)))
    pt = tgt_emb.init(rng, jnp.zeros((4, 90), jnp.int32))
    params = {"ctx": pc, "pred": pp, "tgt": pt}
    opt = optax.adamw(args.lr, weight_decay=1e-4); ost = opt.init(params)

    def vicreg(a, b, w):
        # masked (weight w in {0,1}) invariance + variance (anti-collapse) + covariance,
        # via weighted statistics so no boolean indexing is needed under jit.
        W = w.sum() + 1e-6
        inv = (((a - b) ** 2).mean(-1) * w).sum() / W
        def stats(z):
            mu = (z * w[:, None]).sum(0) / W
            zc = (z - mu) * w[:, None]
            var = (zc ** 2).sum(0) / W
            vloss = jnp.mean(nn.relu(1 - jnp.sqrt(var + 1e-4)))
            cov = (zc.T @ zc) / W
            off = cov - jnp.diag(jnp.diag(cov))
            return vloss, (off ** 2).sum() / z.shape[1]
        va, ca = stats(a); vb, cb = stats(b)
        return inv + 25.0 * (va + vb) + 1.0 * (ca + cb)

    def jloss(params, feats, y, valid):
        z = ctx.apply(params["ctx"], *feats)          # (B,T,DM) causal context THROUGH t (incl PA t matchup token, not its outcome)
        # predict the latent embedding of PA t's own outcome from context z[t] (same target as MLP/transformer)
        vt = valid & (y >= 0)
        phat = pred.apply(params["pred"], z)
        ytgt = tgt_emb.apply(params["tgt"], jnp.clip(y, 0, NOUT - 1))
        ytgt = jax.lax.stop_gradient(ytgt)            # BYOL-style stop-grad on target
        w = vt.reshape(-1).astype(jnp.float32)
        return vicreg(phat.reshape(-1, DM), ytgt.reshape(-1, DM), w)

    @jax.jit
    def jstep(params, ost, feats, y, valid):
        l, g = jax.value_and_grad(jloss)(params, feats, y, valid)
        upd, ost = opt.update(g, ost, params); return optax.apply_updates(params, upd), ost, l

    def bf(d, idx):
        return (jnp.array(d["bat"][idx]), jnp.array(d["pit"][idx]), jnp.array(d["park"][idx]),
                jnp.array(d["bat_rate"][idx]), jnp.array(d["pit_rate"][idx]), jnp.array(d["state"][idx]))

    rng2 = np.random.default_rng(args.seed); G = tr["y"].shape[0]; t0 = time.time()
    print("JEPA pretrain...", flush=True)
    for i in range(args.steps):
        idx = rng2.integers(0, G, args.batch)
        params, ost, l = jstep(params, ost, bf(tr, idx), jnp.array(tr["y"][idx]),
                               jnp.array(tr["valid"][idx], bool))
        if i % 500 == 0:
            print(f"  step {i}  jepa-loss {float(l):.4f}  {time.time()-t0:.0f}s", flush=True)

    # linear probe: frozen context repr -> outcome logits (train probe on 2015-22)
    print("Linear probe on frozen features...", flush=True)
    def feats_repr(d, idx):
        return np.array(ctx.apply(params["ctx"], *bf(d, idx)))
    W = np.zeros((DM, NOUT)); blin = np.zeros(NOUT)
    Wj = jnp.array(W); bj = jnp.array(blin)
    popt = optax.adam(1e-2); pst = popt.init((Wj, bj))
    def ploss(Wb, z, y, v):
        W_, b_ = Wb; logits = z @ W_ + b_
        ll = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.clip(y, 0, NOUT - 1))
        return (ll * v).sum() / v.sum()
    pgrad = jax.jit(jax.value_and_grad(ploss))
    for i in range(1500):
        idx = rng2.integers(0, G, args.batch)
        z = jnp.array(feats_repr(tr, idx))
        y = jnp.array(tr["y"][idx]); v = jnp.array(tr["valid"][idx] & (tr["y"][idx] >= 0), jnp.float32)
        l, g = pgrad((Wj, bj), z, y, v); upd, pst = popt.update(g, pst); Wj, bj = optax.apply_updates((Wj, bj), upd)

    # eval probe on test
    nll = n = corr = 0.0
    Gte = te["y"].shape[0]
    bat_all, y_all, p_all = [], [], []
    for i in range(0, Gte, 128):
        idx = np.arange(i, min(i + 128, Gte))
        z = feats_repr(te, idx)
        logits = np.array(z @ Wj + bj); y = te["y"][idx]; v = te["valid"][idx]
        p = np.exp(logits - logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        m = v & (y >= 0); yi = np.clip(y, 0, NOUT - 1)
        nll += -np.log(np.clip(p[m, yi[m]], 1e-7, 1)).sum(); corr += (p[m].argmax(-1) == yi[m]).sum(); n += m.sum()
        bat_all.append(te["bat"][idx].reshape(-1)); y_all.append(y.reshape(-1))
        p_all.append(p.reshape(-1, NOUT))
    print(f"\n=== JEPA (probe) TEST ({int(n)} PAs) ===", flush=True)
    print(f"  per-PA NLL {nll/n:.4f}   accuracy {corr/n:.4f}   (production ~1.52)", flush=True)
    open("data/eval2/seq_jepa.txt", "a").write(f"jepa-probe test NLL {nll/n:.4f} acc {corr/n:.4f} n {int(n)}\n")
    tag = args.tag or f"jepa train<={args.train_end}"
    player_corr(np.concatenate(bat_all), np.concatenate(p_all), np.concatenate(y_all),
                tag, "data/eval2/arch_newfeatures.txt")


if __name__ == "__main__":
    main()
