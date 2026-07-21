"""Measure the production SVI model (v12) on the SAME per-batter rate-correlation
metric as wm_sweep, for an apples-to-apples comparison against the best
discriminative config. v12 is conditioned (real matchups) with its b_heur recal;
per batter we accumulate expected P(K/BB/hit/HR) and correlate with real rates.
"""
from __future__ import annotations
import numpy as np, polars as pl, pickle
from functools import partial
from diamondworldjax.paths import processed_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.data.pa_batching import build_pa_batch
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index, apply_park_idx
from diamondworldjax.model.pa_model import pa_model
from diamondworldjax.sim.rules_engine import PA_OUTCOME_IDX

KIDX, HRIDX = PA_OUTCOME_IDX["K"], PA_OUTCOME_IDX["HR"]
BB_IDX = [PA_OUTCOME_IDX["BB"], PA_OUTCOME_IDX["HBP"]]
HIT_IDX = [PA_OUTCOME_IDX[x] for x in ("1B", "2B", "3B", "HR")]
TRAIN = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/dwjax_pa_v12/dwjax_step_0050000.pkl")
    ap.add_argument("--recal", default="data/eval2/v12_cal_params.npz")
    ap.add_argument("--recency-halflife", type=float, default=2.0)
    ap.add_argument("--skill-mode", choices=["prior", "mean"], default="mean",
                    help="mean substitutes the learned player_mu (correct for a non-collapsed "
                         "latent); prior samples N(0,1) (legacy).")
    ap.add_argument("--tag", default="v12")
    ap.add_argument("--train-end", type=int, default=2022,
                    help="Last training season for the player table (must match the checkpoint's "
                         "--train-end). 2023 folds in the previous season; then test 2024 only.")
    ap.add_argument("--test-seasons", default="2023,2024",
                    help="Comma list of eval seasons. Use 2024 when the table includes 2023.")
    ap.add_argument("--mle", default=None,
                    help="Path to mle_rates.npz (ids, rates=[hit,bb,k,hr]); injects translated "
                         "minor-league rate features for rookies unseen in training, de-blanking "
                         "them instead of collapsing to the shared unknown slot.")
    args = ap.parse_args()
    TRAIN = list(range(2015, args.train_end + 1))
    test_seasons = [int(x) for x in args.test_seasons.split(",")]
    import jax, jax.numpy as jnp, numpyro.handlers as nh
    params = pickle.load(open(args.ckpt, "rb"))["params"]
    if args.skill_mode == "mean" and "player_mu" in params:
        params = {**params, "player_skills": params["player_mu"]}
    b_heur = np.load(args.recal)["b_heur"].astype(np.float64)
    trp = load_seasons(TRAIN, data_root=processed_root())
    ptab = _build_player_table(trp, recency_halflife=args.recency_halflife)
    park_map = _build_park_index(trp); id2i = ptab["id_to_idx"]; del trp

    n_rookie = 0
    if args.mle:
        # Append rookies (unseen in training) with translated minor-league rate features.
        # New index per rookie; the skill latent gets the prior mean (0) since we have no
        # MLB posterior for them, so the prediction rides on the MLE rate features.
        mle = np.load(args.mle); rids = mle["ids"].astype(int); rrates = mle["rates"]
        new = [(rid, rr) for rid, rr in zip(rids, rrates) if int(rid) not in id2i]
        if new:
            P0 = ptab["stats"].shape[0]; F = ptab["stats"].shape[1]
            add = np.zeros((len(new), F), np.float32)
            for j, (_, rr) in enumerate(new):
                add[j, 0], add[j, 1], add[j, 2], add[j, 3] = rr  # hit, bb, k, hr rates
                add[j, 4] = 150.0                                # nominal PA weight
            ptab["stats"] = np.concatenate([ptab["stats"], add], 0)
            ptab["league"] = np.concatenate([ptab["league"], np.zeros(len(new), np.int32)])
            ptab["hand"] = np.concatenate([ptab["hand"], np.zeros(len(new), np.int32)])
            for hk in ("bat_hand", "pit_hand"):
                ptab[hk] = np.concatenate([ptab[hk], np.full(len(new), 0.5, np.float32)])
            sk = np.asarray(params["player_skills"])
            params = {**params, "player_skills": np.concatenate(
                [sk, np.zeros((len(new), sk.shape[1]), sk.dtype)], 0)}
            for j, (rid, _) in enumerate(new):
                id2i[int(rid)] = P0 + j
            n_rookie = len(new)

    te = load_seasons(test_seasons, data_root=processed_root()).filter(
        pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
    te = apply_park_idx(te, park_map)
    pt = {"stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
          "hand": jnp.array(ptab["hand"]),
          "bat_hand": jnp.array(ptab.get("bat_hand", np.full(len(ptab["hand"]), .5, np.float32))),
          "pit_hand": jnp.array(ptab.get("pit_hand", np.full(len(ptab["hand"]), .5, np.float32)))}
    model_fn = partial(pa_model, outcome_only=True, fatigue=True)
    gids = te["game_pk"].unique().to_numpy()

    # per-batter accumulators
    P = len(ptab["hand"])
    sumK = np.zeros(P); sumBB = np.zeros(P); sumHit = np.zeros(P); sumHR = np.zeros(P)
    rK = np.zeros(P); rBB = np.zeros(P); rHit = np.zeros(P); rHR = np.zeros(P); cnt = np.zeros(P)
    for i in range(0, len(gids), 64):
        df = te.filter(pl.col("game_pk").is_in(gids[i:i+64].tolist())).sort(["game_pk", "at_bat_number"])
        b = build_pa_batch(df)
        batmap = np.vectorize(lambda x: id2i.get(int(x), 0))(np.array(b["batter_ids"]))
        for k in ("pitcher_ids", "batter_ids"):
            b[k] = np.vectorize(lambda x: id2i.get(int(x), 0))(np.array(b[k]))
        for k in list(b):
            if k != "game_ids": b[k] = jnp.array(np.array(b[k]))
        with nh.seed(rng_seed=0):
            with nh.substitute(data=params):
                with nh.trace() as tr:
                    model_fn(b, pt, teacher_force=False)
        lg = np.array(tr["pa_outcome"]["fn"].logits) + b_heur
        pp = np.exp(lg - lg.max(-1, keepdims=True)); pp /= pp.sum(-1, keepdims=True)
        valid = np.array(b["pa_valid"]); y = np.array(b["pa_outcome"])
        m = valid & (y >= 0)
        bidx = batmap[m]; pv = pp[m]; yi = np.clip(y[m], 0, 8)
        np.add.at(sumK, bidx, pv[:, KIDX]); np.add.at(sumHR, bidx, pv[:, HRIDX])
        np.add.at(sumBB, bidx, pv[:, BB_IDX].sum(1)); np.add.at(sumHit, bidx, pv[:, HIT_IDX].sum(1))
        np.add.at(rK, bidx, (yi == KIDX)); np.add.at(rHR, bidx, (yi == HRIDX))
        np.add.at(rBB, bidx, np.isin(yi, BB_IDX)); np.add.at(rHit, bidx, np.isin(yi, HIT_IDX))
        np.add.at(cnt, bidx, 1.0)

    keep = cnt >= 150
    def corr(s, r): return float(np.corrcoef((s[keep] / cnt[keep]), (r[keep] / cnt[keep]))[0, 1])
    cK, cBB, cHit, cHR = corr(sumK, rK), corr(sumBB, rBB), corr(sumHit, rHit), corr(sumHR, rHR)
    rookie_kept = int((keep[-n_rookie:]).sum()) if n_rookie else 0
    line = (f"{args.tag} (SVI, conditioned+recal, skill={args.skill_mode}, "
            f"train<= {args.train_end}, test {args.test_seasons}, mle={bool(args.mle)}) | "
            f"corr K {cK:.3f} BB {cBB:.3f} Hit {cHit:.3f} HR {cHR:.3f} AVG {np.mean([cK,cBB,cHit,cHR]):.3f} "
            f"(np={int(keep.sum())}, rookies_injected={n_rookie}, rookies_kept={rookie_kept})")
    print(line)
    open(f"data/eval2/prod_playercorr_{args.tag}.txt", "w").write(line + "\n")
    # save per-batter predicted+real sums (indexed by player idx) for the hybrid
    np.savez(f"data/eval2/prod_rates_{args.tag}.npz", sumK=sumK, sumBB=sumBB, sumHit=sumHit, sumHR=sumHR,
             rK=rK, rBB=rBB, rHit=rHit, rHR=rHR, cnt=cnt)


if __name__ == "__main__":
    main()
