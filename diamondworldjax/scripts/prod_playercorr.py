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
    import jax, jax.numpy as jnp, numpyro.handlers as nh
    V12 = "checkpoints/dwjax_pa_v12/dwjax_step_0050000.pkl"
    params = pickle.load(open(V12, "rb"))["params"]
    b_heur = np.load("data/eval2/v12_cal_params.npz")["b_heur"].astype(np.float64)
    trp = load_seasons(TRAIN, data_root=processed_root())
    ptab = _build_player_table(trp, recency_halflife=2.0)
    park_map = _build_park_index(trp); id2i = ptab["id_to_idx"]; del trp
    te = load_seasons([2023, 2024], data_root=processed_root()).filter(
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
    line = (f"PRODUCTION v12 (SVI, conditioned+recal) same metric | "
            f"corr K {cK:.3f} BB {cBB:.3f} Hit {cHit:.3f} HR {cHR:.3f} AVG {np.mean([cK,cBB,cHit,cHR]):.3f} "
            f"(np={int(keep.sum())})")
    print(line)
    open("data/eval2/prod_playercorr.txt", "w").write(line + "\n")
    # save per-batter predicted+real sums (indexed by player idx) for the hybrid
    np.savez("data/eval2/prod_rates.npz", sumK=sumK, sumBB=sumBB, sumHit=sumHit, sumHR=sumHR,
             rK=rK, rBB=rBB, rHit=rHit, rHR=rHR, cnt=cnt)


if __name__ == "__main__":
    main()
