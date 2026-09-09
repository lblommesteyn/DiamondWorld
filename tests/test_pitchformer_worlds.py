"""World policy and six-model launcher regressions."""
from pathlib import Path
import pickle
import shutil
import subprocess

import pytest

from diamondworldjax.eval.pitchformer_worlds import PitchformerWorlds
from diamondworldjax.simulate.pitchformer_rollout import PitchformerHeads


def test_world_sampling_and_calibration_are_separate(tmp_path, monkeypatch):
    import diamondworldjax.model.bayesian_pitchformer as native
    calls = []
    def export(checkpoint, seed, sample):
        calls.append((seed, sample))
        return {h: (h, seed, sample) for h in 'abcd'}
    monkeypatch.setattr(native, 'export_world', export)
    with (tmp_path / 'bayesian_test.pkl').open('wb') as f:
        pickle.dump({}, f)
    base = PitchformerHeads('a', 'b', 'c', 'd', 'a0', 'b0', 'c0', 'd0')
    worlds = PitchformerWorlds(base, tmp_path, 'test', {'bayesian': True}, seed=42)
    rep0, rep1, repeat = worlds.for_rep(0), worlds.for_rep(1), worlds.for_rep(0)
    assert rep0.a_params == repeat.a_params
    assert rep0.a_params != rep1.a_params
    assert rep0.a_params[1] == rep0.d_params[1]
    assert all(sample for _, sample in calls)
    # The evaluator gets one object to reuse throughout each rep; base unmodified.
    assert base.a_params == 'a0'
    assert rep0.a is base.a
    mean = worlds.calibration_heads()
    assert mean.a_params[2] is False
    assert worlds.calibration_heads() is mean
    assert len(calls) == 4
    assert worlds.report(2)['skill_seeds'] == [rep0.a_params[1], rep1.a_params[1]]
    fixed_mean = PitchformerWorlds(base, tmp_path, 'test', {'bayesian': True}, mode='mean')
    assert fixed_mean.for_rep(0) is fixed_mean.for_rep(1)


def test_fixed_and_exported_worlds_and_missing_posterior(tmp_path):
    base = PitchformerHeads(None, None, None, None, {}, {})
    for meta in [None, {'bayesian': False}, {'bayesian': True, 'posterior_world':
                {'source_tag': 'original', 'mode': 'sample', 'seed': 7}}]:
        worlds = PitchformerWorlds(base, tmp_path, 'test', meta)
        assert worlds.for_rep(0) is base and worlds.for_rep(2) is base
        assert not worlds.report(3)['resample_per_rep']
    with pytest.raises(ValueError, match='requires a native Bayesian'):
        PitchformerWorlds(base, tmp_path, 'test', None, mode='sample')
    with pytest.raises(ValueError, match='Posterior required'):
        PitchformerWorlds(base, tmp_path, 'test', {'bayesian': True})


def bash():
    result = shutil.which('bash')
    if not result:
        pytest.skip('Bash not installed')
    return result


def test_six_model_dry_run_and_selection():
    root = Path(__file__).resolve().parents[1]
    command = [bash(), 'scripts/run_six_models.sh', '--dry-run', '--sim-reps', '2']
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    output = result.stdout
    assert output.count('[train]') == 6
    for flag in ['--history-reset game', '--dropout 0.1', '--position-encoding sinusoidal',
                 '--window-size 32', '--context-len 32', '--missing-samples 2', '--c-event-mode bundles']:
        assert output.count(flag) == 3
    custom_abcd = subprocess.run(command + ['--models', 'abcd_none', '--abcd-window', '8',
        '--abcd-max-len', '48', '--game-batch', '4', '--missing-samples', '3',
        '--history-reset', 'half_inning', '--abcd-dropout', '0.2', '--limit-games', '0'],
        cwd=root, capture_output=True, text=True, check=True).stdout
    for flag in ['--window-size 8', '--context-len 8', '--max-len 48', '--batch-games 4',
                 '--missing-samples 3', '--history-reset half_inning', '--dropout 0.2', '--limit-games 0']:
        assert flag in custom_abcd
    for options in [['--abcd-dropout', '1'], ['--history-reset', 'bad'], ['--abcd-window', '160'],
                    ['--missing-samples', '0'], ['--game-batch', '0'], ['--skill-feature-mode', 'bad']]:
        bad = subprocess.run(command + options, cwd=root, capture_output=True, text=True)
        assert bad.returncode != 0 and '[train]' not in bad.stdout
    assert output.count('--ss-rate 0 ') == 3
    assert output.count('[games_0]') == 3 and output.count('[games_1]') == 3
    assert output.count('[games]') == 3
    assert '--pa-arch gru' in output and '--pa-arch transformer' in output
    assert '--player-skills bayesian' in output and '--player-skills none' in output and '--player-skills id' in output
    assert '--train-end 2023' in output and '2020' in output
    result = subprocess.run(command + ['--models', 'abcd_bayesian', '--skill-mode', 'mean',
                            '--skill-features', 'features with spaces.npz'], cwd=root,
                            capture_output=True, text=True, check=True)
    assert result.stdout.count('[train]') == 1
    assert '--skill-mode mean' in result.stdout
    assert '--skill-features' in result.stdout
    for flag in ['--pa-ss-rate', '--ss-rate']:
        custom = subprocess.run(command + ['--models', 'pa_gru', flag, '0.25'], cwd=root,
                                capture_output=True, text=True, check=True)
        assert '--ss-rate 0.25 ' in custom.stdout
    for rate in ['-0.1', '1.1', 'nan', 'abc']:
        bad = subprocess.run(command + ['--pa-ss-rate', rate], cwd=root, capture_output=True, text=True)
        assert bad.returncode != 0 and '[train]' not in bad.stdout
    for options in [['--models', 'bad'], ['--models', 'pa,pa'], ['--sim-reps', '0'], ['--abcd-steps', '1']]:
        result = subprocess.run(command + options, cwd=root, capture_output=True, text=True)
        assert result.returncode != 0
        assert '[train]' not in result.stdout
