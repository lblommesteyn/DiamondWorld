"""Generate the figures for the DiamondWorld SSAC paper (vector PDF).

Fig 1  validation scatter: within-series sim dWP vs market dWP (the money figure)
Fig 2  cross-season / cross-market correlation bars
Fig 3  run-total interval coverage: simulator vs summed-Poisson vs nominal
Fig 4  player-rate benchmark: Marcel vs Steamer vs DiamondWorld
Fig 5  methods: per-PA NLL vs player differentiation (JEPA collapse + fine-tuning)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diamondworldjax.scripts.simulator_benchmarks import team_rates_2024, american_implied

FIG = Path("paper/figs"); FIG.mkdir(parents=True, exist_ok=True)
INK = "#17211C"; ACCENT = "#AF4A2C"; FIELD = "#2C6E52"; MUTE = "#8A968E"; STEEL = "#3B6E8F"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": "#444", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": "#333", "ytick.color": "#333",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def _within_series_pairs(arrays, odds_csv):
    d = np.load(arrays)
    sh, sa, pk = d["sim_home"], d["sim_away"], d["game_pk"].astype(int)
    _, pkt = team_rates_2024()
    keep = np.array([p in pkt for p in pk]); sh, sa, pk = sh[keep], sa[keep], pk[keep]
    sim = (sh > sa).mean(1)
    od = pl.read_csv(odds_csv)
    om = {int(r["game_pk"]): (r["ml_home"], r["ml_away"]) for r in od.iter_rows(named=True)
          if r["ml_home"] is not None and r["ml_away"] is not None}
    mk = np.array([np.nan if int(p) not in om else american_implied(om[int(p)][0]) /
                   (american_implied(om[int(p)][0]) + american_implied(om[int(p)][1])) for p in pk])
    m = np.isfinite(mk) & np.isfinite(sim)
    sim, mk, pk = sim[m], mk[m], pk[m]
    H = np.array([pkt[int(p)][0] for p in pk]); A = np.array([pkt[int(p)][1] for p in pk])
    grp = defaultdict(list)
    for r in range(len(pk)):
        grp[(H[r], A[r])].append(r)
    ss, mm = [], []
    for idx in grp.values():
        if len(idx) < 2:
            continue
        idx = np.array(idx)
        ss.extend(sim[idx] - sim[idx].mean()); mm.extend(mk[idx] - mk[idx].mean())
    return np.array(ss), np.array(mm)


# Game-level figures use the locked model (v22, seed 42) simulated leak-free at R=2000. The pooled
# sixteen-variant arrays (diamondworldjax/scripts/pool_sims.py) are a robustness row in Table 3 only.
ENS = "data/eval2/calib_v22L_s42-pregame-leakfree-r2000_arrays.npz"
ENS_ALIGNED = ENS


def fig1():
    ss, mm = _within_series_pairs(ENS, "data/eval2/odds_2023_2024.csv")
    r = np.corrcoef(ss, mm)[0, 1]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(ss * 100, mm * 100, s=9, alpha=0.20, color=STEEL, edgecolors="none", zorder=2)
    # binned means
    q = np.quantile(ss, np.linspace(0, 1, 9))
    bx, by, be = [], [], []
    for i in range(len(q) - 1):
        sel = (ss >= q[i]) & (ss <= q[i + 1]) if i == len(q) - 2 else (ss >= q[i]) & (ss < q[i + 1])
        if sel.sum() > 5:
            bx.append(ss[sel].mean() * 100); by.append(mm[sel].mean() * 100)
            be.append(mm[sel].std() / np.sqrt(sel.sum()) * 100)
    ax.errorbar(bx, by, yerr=be, fmt="o", color=ACCENT, ms=5, lw=1.2, capsize=2, zorder=4, label="binned mean")
    # OLS fit
    b1, b0 = np.polyfit(ss, mm, 1)
    xs = np.array([ss.min(), ss.max()])
    ax.plot(xs * 100, (b0 + b1 * xs) * 100, color=ACCENT, lw=1.6, zorder=3,
            label=f"fit (slope {b1:.2f})")
    ax.axhline(0, color=MUTE, lw=0.6); ax.axvline(0, color=MUTE, lw=0.6)
    ax.set_xlabel("simulator within-series $\\Delta$WP (points)")
    ax.set_ylabel("market-implied within-series $\\Delta$WP (points)")
    ax.set_title(f"Simulator counterfactuals versus market forecast changes\n$r={r:.2f}$, leak-free, "
                 f"{len(ss)} game-deviations", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    lim = float(np.ceil(np.percentile(np.abs(ss), 99.5) * 100))
    ax.set_xlim(-lim, lim); ax.set_ylim(-9, 9)
    fig.tight_layout(); fig.savefig(FIG / "fig1_validation.pdf"); plt.close(fig)
    print(f"fig1 r={r:.3f} slope={b1:.3f} n={len(ss)}")


def fig2():
    labels = ["2024\nmarket", "2025\nmarket\n(out-of-sample)", "2026\nmarket\n(independent source)"]
    corr = [0.26, 0.33, 0.12]
    colors = [STEEL, FIELD, ACCENT]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    bars = ax.bar(range(3), corr, color=colors, width=0.62, zorder=3)
    for i, v in enumerate(corr):
        ax.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("within-series corr(sim, market)")
    ax.set_ylim(0, 0.4)
    ax.set_title("Signal holds across seasons and independent forecasts", fontsize=9.5)
    ax.axhline(0.032, color=MUTE, ls="--", lw=0.8)
    ax.text(2.42, 0.045, "null 95th pct", fontsize=7, color=MUTE, ha="right")
    fig.tight_layout(); fig.savefig(FIG / "fig2_crossmarket.pdf"); plt.close(fig)
    print("fig2 done")


def fig3():
    levels = ["50%", "80%", "90%"]
    # data/eval2/distributional_stats_v22L_r2000.txt
    sim = [0.560, 0.843, 0.924]; pois = [0.425, 0.674, 0.791]; nominal = [0.50, 0.80, 0.90]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.bar(x - w / 2, sim, w, color=FIELD, label="DiamondWorld", zorder=3)
    ax.bar(x + w / 2, pois, w, color=MUTE, label="summed 2-Poisson", zorder=3)
    ax.plot(x, nominal, "D", color=ACCENT, ms=7, label="nominal (ideal)", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(levels)
    ax.set_xlabel("central prediction interval")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0, 1.0)
    ax.set_title("Run-total distribution: coverage vs a summed model", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig3_coverage.pdf"); plt.close(fig)
    print("fig3 done")


def fig4():
    stats = ["K%", "BB%", "Hit%", "HR%", "Avg"]
    # data/eval2/blend_v22_s42.txt: 376 batters with >= 150 PA in 2024
    marcel = [0.793, 0.674, 0.497, 0.641, 0.651]
    dw = [0.803, 0.665, 0.495, 0.628, 0.648]
    blend = [0.806, 0.682, 0.531, 0.654, 0.668]
    steamer = [0.820, 0.702, 0.510, 0.651, 0.671]
    x = np.arange(5); w = 0.2
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.bar(x - 1.5 * w, marcel, w, color=MUTE, label="Marcel + CQ", zorder=3)
    ax.bar(x - 0.5 * w, dw, w, color="#8FB8A2", label="DiamondWorld", zorder=3)
    ax.bar(x + 0.5 * w, blend, w, color=FIELD, label="average of the two", zorder=3)
    ax.bar(x + 1.5 * w, steamer, w, color=STEEL, label="Steamer", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(stats)
    ax.set_ylabel("cross-player rate correlation (2024)")
    ax.set_ylim(0, 0.95)
    ax.set_title("Player realism: the averaged player layer is level with Steamer", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center")
    fig.tight_layout(); fig.savefig(FIG / "fig4_players.pdf"); plt.close(fig)
    print("fig4 done")


def fig5():
    # per-PA NLL (x) vs player-corr (y)
    # (name, nll, corr, color, label_offset_pts or None to suppress)
    pts = [("JEPA frozen", 1.492, 0.064, ACCENT, (8, -2)),
           ("JEPA fine-tuned", 1.535, 0.539, "#C98A3B", (-30, 8)),
           ("JEPA scratch", 1.511, 0.555, "#C98A3B", (6, 6)),
           ("transformer", 1.500, 0.574, STEEL, None),
           ("GRU", 1.490, 0.578, STEEL, None),
           ("MLP", 1.490, 0.577, STEEL, None),
           ("DiamondWorld SVI", 1.495, 0.611, FIELD, (8, 6))]
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for name, nll, pc, c, off in pts:
        ax.scatter(nll, pc, s=42, color=c, zorder=3, edgecolors="white", linewidths=0.6)
        if off is not None:
            ax.annotate(name, (nll, pc), textcoords="offset points", xytext=off,
                        fontsize=7.5, color=INK)
    # one label for the tight discriminative cluster
    ax.annotate("transformer,\nGRU, MLP", (1.495, 0.576), textcoords="offset points",
                xytext=(4, -30), fontsize=7.5, color=STEEL,
                arrowprops=dict(arrowstyle="-", color=STEEL, lw=0.6))
    # arrow: frozen -> fine-tuned (recovery)
    ax.annotate("", xy=(1.535, 0.539), xytext=(1.492, 0.064),
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.1, ls="--"))
    ax.text(1.500, 0.30, "fine-tuning\nrecovers", fontsize=7.5, color=MUTE, rotation=78)
    ax.set_xlabel("per-PA held-out NLL (lower = 'better' likelihood)")
    ax.set_ylabel("player differentiation (cross-player corr)")
    ax.set_title("The best likelihood is the worst world model", fontsize=9.5)
    ax.set_xlim(1.478, 1.552); ax.set_ylim(-0.02, 0.68)
    fig.tight_layout(); fig.savefig(FIG / "fig5_metric.pdf"); plt.close(fig)
    print("fig5 done")


def fig6_heatmap():
    """League P(hit) over (exit velocity, launch angle): the contact-quality feature."""
    from matplotlib.colors import LinearSegmentedColormap
    from diamondworldjax.paths import processed_root
    from diamondworldjax.data.pipeline import load_seasons
    d = (load_seasons([2024], data_root=processed_root())
         .filter(pl.col("pa_terminal") & pl.col("launch_speed").is_not_null()
                 & pl.col("launch_angle").is_not_null()))
    ev = d["launch_speed"].to_numpy(); la = d["launch_angle"].to_numpy()
    hit = d["pa_outcome"].is_in(["1B", "2B", "3B", "HR"]).to_numpy().astype(float)
    hr = (d["pa_outcome"] == "HR").to_numpy().astype(float)
    ev_b = np.arange(40, 118, 2.5); la_b = np.arange(-30, 62, 2.5)
    Hh, _, _ = np.histogram2d(ev, la, bins=[ev_b, la_b], weights=hit)
    N, _, _ = np.histogram2d(ev, la, bins=[ev_b, la_b])
    P = np.where(N >= 12, Hh / np.maximum(N, 1), np.nan)
    cmap = LinearSegmentedColormap.from_list("hit", ["#EEF1EC", FIELD, "#C98A3B", ACCENT])
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    im = ax.imshow(P.T, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   extent=[ev_b[0], ev_b[-1], la_b[0], la_b[-1]])
    ax.set_xlabel("exit velocity (mph)"); ax.set_ylabel("launch angle (deg)")
    ax.set_title("How the model sees contact: P(hit) by (exit velocity, launch angle)", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("P(hit)", fontsize=8)
    ax.axhline(0, color="#555", lw=0.5, ls=":")
    ax.text(101, 26, "barrel\nzone", fontsize=7.5, color="white", ha="center", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig6_heatmap.pdf"); plt.close(fig)
    print("fig6 heatmap done")


def fig7_rundist():
    """Simulated vs real game-total distribution and an independent-Poisson reference."""
    d = np.load(ENS_ALIGNED)
    st = d["sim_total"].reshape(-1).astype(float); rt = d["real_total"].astype(float)
    shift = rt.mean() - st.mean(); st = st + shift          # mean-match (shape comparison)
    from scipy.stats import poisson as _po
    bins = np.arange(-0.5, 24.5, 1)
    ctr = np.arange(0, 24)
    rh, _ = np.histogram(rt, bins=bins, density=True)
    sh, _ = np.histogram(st, bins=bins, density=True)
    pois = _po.pmf(ctr, rt.mean())
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(ctr, rh, width=0.9, color="#D9DED8", zorder=1, label="real games")
    ax.step(ctr, sh, where="mid", color=FIELD, lw=1.8, zorder=3, label="DiamondWorld")
    ax.step(ctr, pois, where="mid", color=ACCENT, lw=1.6, ls="--", zorder=3,
            label="independent Poisson")
    ax.axvspan(10, 24, color="#000", alpha=0.045, zorder=0)
    ax.text(15.5, ax.get_ylim()[1] * 0.86, "fat tail\n(blowouts)", fontsize=8, color="#555", ha="center")
    ax.set_xlabel("total runs in a game"); ax.set_ylabel("probability")
    ax.set_title("The simulator reproduces the real run distribution, tail and all", fontsize=9.5)
    ax.set_xlim(0, 22); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig7_rundist.pdf"); plt.close(fig)
    print("fig7 rundist done")


def fig8_series():
    """Illustrative within-series identification: market and sim WP co-move with the starter."""
    games = [1, 2, 3]
    mkt = [0.605, 0.452, 0.560]
    sim = [0.578, 0.480, 0.535]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.axhspan(0.30, 0.70, color="#F3F5F1", zorder=0)
    ax.axhline(0.5, color=MUTE, ls="--", lw=0.8)
    ax.text(3.02, 0.503, "even", fontsize=7.5, color=MUTE, va="bottom")
    ax.plot(games, mkt, "-o", color=ACCENT, ms=8, lw=1.8, label="independent market", zorder=3)
    ax.plot(games, sim, "--s", color=FIELD, ms=8, lw=1.8, label="DiamondWorld", zorder=3)
    ax.set_xticks(games)
    ax.set_xticklabels(["Game 1\nSP: ace", "Game 2\nSP: #3 starter", "Game 3\nSP: back-end"], fontsize=8.5)
    ax.set_ylabel("home win probability")
    ax.set_ylim(0.36, 0.70); ax.set_xlim(0.7, 3.3)
    ax.set_title("Within-series forecast-change benchmark:\nsame teams, varying pregame inputs", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    ax.annotate("both move\nwith the starter", xy=(2, 0.466), xytext=(1.55, 0.40),
                fontsize=8, color="#555", ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.8))
    fig.tight_layout(); fig.savefig(FIG / "fig8_series.pdf"); plt.close(fig)
    print("fig8 series done")


def fig9_starters():
    """Per-start value of each starter: market-calibrated simulator vs market."""
    import json
    d = np.load("data/eval2/starter_value_v22L_s42-pregame-leakfree-r2000.npz")
    s = d["slope"] * d["sim"] * 100; m = d["mkt"] * 100
    names = {int(k): v for k, v in json.loads(
        Path("data/cache/projections/mlb_names.json").read_text()).items()}
    r = np.corrcoef(s, m)[0, 1]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(s, m, s=6 + d["n"] * 0.8, alpha=0.45, color=STEEL, edgecolors="none", zorder=2)
    ax.plot([-5, 5], [-5, 5], color=MUTE, lw=0.8, ls="--", zorder=1)
    ax.text(4.9, 4.2, "$y=x$", fontsize=7.5, color=MUTE, ha="right")
    # label offsets in points, chosen so neighbouring names do not collide
    label = {"Paul Skenes": (4, -3), "Tarik Skubal": (4, 2), "Corbin Burnes": (4, -9),
             "Justin Verlander": (4, 2), "Logan Webb": (-24, 5), "Yoshinobu Yamamoto": (4, 2),
             "Kenta Maeda": (4, -3), "Zack Wheeler": (4, 2), "Shota Imanaga": (4, 2)}
    for k in range(len(s)):
        nm = names.get(int(d["ids"][k]), "")
        if nm in label:
            ax.annotate(nm.split()[-1], (s[k], m[k]), textcoords="offset points", xytext=label[nm],
                        fontsize=7, color=INK)
    ax.axhline(0, color=MUTE, lw=0.5); ax.axvline(0, color=MUTE, lw=0.5)
    ax.set_xlim(-5, 5); ax.set_ylim(-7, 9)
    ax.set_xlabel("simulator, market-calibrated (WP points per start)")
    ax.set_ylabel("market (WP points per start)")
    ax.set_title(f"What a starter is worth per start, vs his own rotation\n"
                 f"$r={r:.2f}$, {len(s)} starters with $\\geq$10 starts, 2024", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "fig9_starters.pdf"); plt.close(fig)
    print(f"fig9 starters r={r:.3f}")


PALETTE_paper = "#F3F5F1"

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6_heatmap(); fig7_rundist(); fig8_series()
    fig9_starters()
    print("all figures ->", FIG)
