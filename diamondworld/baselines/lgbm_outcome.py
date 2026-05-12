from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl
import lightgbm as lgb

from diamondworld.baselines.markov_re24 import MarkovRE24Simulator
from diamondworld.baselines.base import PA_OUTCOMES, PA_OUTCOME_IDX

# Feature columns used by the LightGBM model (all from pa_terminal rows)
_NUMERIC_FEATURES = [
    "balls", "strikes", "outs", "base_state", "score_diff",
    "inning", "tto", "tracking_era",
    "plate_x", "plate_z", "release_speed", "pfx_x", "pfx_z",
]
_CATEGORICAL_FEATURES = ["stand", "p_throws", "pitch_type", "park_id"]
_ALL_FEATURES = _NUMERIC_FEATURES + ["stand_enc", "p_throws_enc", "pitch_type_enc", "park_id_enc"]


class LGBMOutcomeSimulator(MarkovRE24Simulator):
    """Baseline B3: MarkovRE24 with LightGBM outcome model.

    Overrides the outcome sampling with a multiclass LightGBM model trained on
    pitch-level features at PA-terminal rows. Transitions and simulation loop
    are inherited from MarkovRE24Simulator.
    """

    def fit(self, pitches: pl.DataFrame) -> None:
        # Fit Markov transitions and stand/p_throws distributions first
        super().fit(pitches)

        # Build pitch feature pool for simulation (all terminal rows with complete features)
        terminal = pitches.filter(pl.col("pa_terminal"))
        self._build_pitch_pool(terminal)

        # Fit LightGBM on training data (non-null pa_outcome), val on season 2022
        self._fit_lgbm(terminal)

    def _encode_stand(self, v: str | None) -> int:
        return {"L": 0, "R": 1}.get(v or "R", 1)

    def _encode_p_throws(self, v: str | None) -> int:
        return {"L": 0, "R": 1}.get(v or "R", 1)

    def _fit_encoders(self, terminal: pl.DataFrame) -> None:
        """Fit label encoders for pitch_type and park_id."""
        pitch_types = sorted(
            {v for v in terminal["pitch_type"].drop_nulls().to_list() if v is not None}
        )
        park_ids = sorted(
            {v for v in terminal["park_id"].drop_nulls().to_list() if v is not None}
        )
        self._pitch_type_enc: dict[str, int] = {pt: i for i, pt in enumerate(pitch_types)}
        self._park_id_enc: dict[str, int] = {pk: i for i, pk in enumerate(park_ids)}

    def _encode_row(self, row: dict[str, Any]) -> list[float]:
        """Build feature vector for one row."""
        feats = []
        for col in _NUMERIC_FEATURES:
            v = row.get(col)
            med = self._medians.get(col, 0.0)
            feats.append(float(v) if v is not None else med)
        # Categorical encodings
        feats.append(float(self._encode_stand(row.get("stand"))))
        feats.append(float(self._encode_p_throws(row.get("p_throws"))))
        pt = row.get("pitch_type")
        feats.append(float(self._pitch_type_enc.get(pt, -1)) if pt else -1.0)
        pk = row.get("park_id")
        feats.append(float(self._park_id_enc.get(pk, -1)) if pk else -1.0)
        return feats

    def _compute_medians_from_df(self, df: pl.DataFrame) -> None:
        """Compute median for each numeric feature from a Polars DataFrame."""
        self._medians: dict[str, float] = {}
        for col in _NUMERIC_FEATURES:
            if col in df.columns:
                series = df[col].drop_nulls()
                self._medians[col] = float(series.median()) if len(series) > 0 else 0.0
            else:
                self._medians[col] = 0.0

    def _compute_medians(self, rows: list[dict[str, Any]]) -> None:
        """Compute median for each numeric feature (for null imputation).
        Falls back to column-by-column iteration from row dicts.
        Prefer _compute_medians_from_df when a DataFrame is available.
        """
        col_vals: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for col in _NUMERIC_FEATURES:
                v = row.get(col)
                if v is not None:
                    col_vals[col].append(float(v))
        self._medians: dict[str, float] = {}
        for col in _NUMERIC_FEATURES:
            vals = col_vals.get(col, [0.0])
            self._medians[col] = float(np.median(vals)) if vals else 0.0

    def _fit_lgbm(self, terminal: pl.DataFrame) -> None:
        """Train LightGBM multiclass model on terminal rows with non-null pa_outcome."""
        labeled = terminal.filter(pl.col("pa_outcome").is_not_null())
        self._fit_encoders(labeled)

        all_rows = labeled.to_dicts()
        self._compute_medians_from_df(labeled)

        # Split by season: train on < 2022, val on == 2022
        train_rows = [r for r in all_rows if (r.get("season") or 0) < 2022]
        val_rows = [r for r in all_rows if r.get("season") == 2022]

        if not train_rows:
            # Fallback: use all rows for training
            train_rows = all_rows
            val_rows = []

        X_train = np.array([self._encode_row(r) for r in train_rows], dtype=np.float32)
        y_train = np.array(
            [PA_OUTCOME_IDX[r["pa_outcome"]] for r in train_rows], dtype=np.int32
        )

        feature_names = _ALL_FEATURES

        train_dataset = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        valid_datasets = []
        if val_rows:
            X_val = np.array([self._encode_row(r) for r in val_rows], dtype=np.float32)
            y_val = np.array(
                [PA_OUTCOME_IDX[r["pa_outcome"]] for r in val_rows], dtype=np.int32
            )
            val_dataset = lgb.Dataset(X_val, label=y_val, feature_name=feature_names)
            valid_datasets = [val_dataset]

        params = {
            "objective": "multiclass",
            "num_class": 8,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "verbose": -1,
            "seed": 42,
            "num_threads": 1,
        }

        if valid_datasets:
            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
            n_rounds = 500
        else:
            # No validation set — cap rounds to avoid over-long training
            callbacks = [lgb.log_evaluation(-1)]
            n_rounds = 200

        self._lgbm_model = lgb.train(
            params,
            train_dataset,
            num_boost_round=n_rounds,
            valid_sets=valid_datasets if valid_datasets else None,
            callbacks=callbacks,
        )

    def _build_pitch_pool(self, terminal: pl.DataFrame) -> None:
        """Build a pool of pitch feature dicts for simulation sampling."""
        feat_cols = _NUMERIC_FEATURES + _CATEGORICAL_FEATURES
        available = [c for c in feat_cols if c in terminal.columns]
        pool_rows = terminal.select(available).to_dicts()
        self._pitch_feat_pool: list[dict[str, Any]] = [
            r for r in pool_rows
            if r.get("plate_x") is not None and r.get("plate_z") is not None
        ]
        if not self._pitch_feat_pool:
            self._pitch_feat_pool = pool_rows  # use all if no complete rows

    def _prefetch_outcomes(self, n: int = 500) -> None:
        """Pre-compute a batch of outcome samples to amortize LGB prediction overhead.

        Samples n random rows from the pitch pool, builds the feature matrix,
        runs a single batched predict, and stores sampled outcomes in a deque.
        """
        from collections import deque

        pool = self._pitch_feat_pool
        if not pool:
            self._outcome_cache: deque[str] = deque()
            return

        # Sample n random rows
        idxs = np.random.randint(len(pool), size=n)
        rows = [pool[i] for i in idxs]

        # Build feature matrix
        X = np.array([self._encode_row(r) for r in rows], dtype=np.float32)
        probs_2d = self._lgbm_model.predict(X)  # shape (n, 8)
        probs_2d = probs_2d / probs_2d.sum(axis=1, keepdims=True)

        sampled = [
            PA_OUTCOMES[int(np.random.choice(8, p=probs_2d[i]))]
            for i in range(n)
        ]
        self._outcome_cache = deque(sampled)

    def _sample_outcome(
        self, bs: int, outs: int, stand: str, p_throws: str
    ) -> str:
        """Override: use LightGBM (batched via prefetch cache)."""
        if not hasattr(self, "_outcome_cache") or len(self._outcome_cache) == 0:
            self._prefetch_outcomes(500)
        return self._outcome_cache.popleft()

    def _simulate_half_inning(self) -> int:
        """Inherited simulation loop (uses overridden _sample_outcome)."""
        return super()._simulate_half_inning()
