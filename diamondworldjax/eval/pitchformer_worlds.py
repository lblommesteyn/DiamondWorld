"""Posterior skill policy shared by the ABCD evaluation entry points."""
from dataclasses import replace
from pathlib import Path
import pickle


class PitchformerWorlds:
    def __init__(self, heads, directory, tag, metadata, mode='auto', seed=0):
        self.base = heads
        self.seed = seed
        self.checkpoint = None
        self._mean = None
        bayesian = bool(metadata and metadata.get('bayesian'))
        exported = (metadata or {}).get('posterior_world')
        self.mode = ('sample' if bayesian and not exported else 'fixed') if mode == 'auto' else mode
        if self.mode == 'sample' and not bayesian:
            raise ValueError('--skill-mode sample requires a native Bayesian ABCD checkpoint')
        if not bayesian:
            self.mode = 'fixed'
        if bayesian and self.mode != 'fixed':
            source = exported['source_tag'] if exported else tag
            path = Path(directory) / f'bayesian_{source}.pkl'
            if not path.exists():
                raise ValueError(f'Posterior required for --skill-mode {self.mode}: {path}')
            with path.open('rb') as f:
                self.checkpoint = pickle.load(f)
            missing = [h for h in self.checkpoint.get('heads', '') if getattr(heads, h) is None]
            if missing:
                raise ValueError(f'Missing standalone head exports for Bayesian posterior: {missing}')
        self.source_world = exported

    def _materialize(self, seed, sample):
        from diamondworldjax.model.bayesian_pitchformer import export_world
        variables = export_world(self.checkpoint, seed=seed, sample=sample)
        return replace(self.base, **{h + '_params': value for h, value in variables.items()})

    def for_rep(self, rep):
        if self.mode == 'sample':
            # Separate stream from outcome RNG; one shared draw across every
            # head, game and batch in this replication, never per pitch.
            return self._materialize((self.seed + rep * 1_000_003 + 2_147_483_647) % 2**32, True)
        return self.calibration_heads()

    def calibration_heads(self):
        if self.checkpoint is None:
            return self.base
        if self._mean is None:
            self._mean = self._materialize(self.seed, False)
        return self._mean

    def report(self, reps):
        return dict(mode=self.mode, resample_per_rep=self.mode == 'sample',
            skill_seeds=[(self.seed + rep * 1_000_003 + 2_147_483_647) % 2**32
                         for rep in range(reps)] if self.mode == 'sample' else [],
            calibration_mode='posterior_mean_skills' if self.checkpoint is not None else 'fixed_checkpoint',
            source_world=self.source_world)
