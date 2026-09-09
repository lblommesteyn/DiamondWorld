"""Export one coherent Bayesian ABCD world for the existing ABCD evaluators."""
import argparse
from copy import deepcopy
from pathlib import Path
import pickle
import jax
from diamondworldjax.model.bayesian_pitchformer import export_world
from diamondworldjax.model.pitchformer_checkpoint import load_metadata, save_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params-dir', required=True)
    parser.add_argument('--tag', required=True, help='Source Bayesian training tag')
    parser.add_argument('--out-tag', required=True, help='Distinct tag for the materialized world')
    parser.add_argument('--mode', choices=['mean', 'sample'], default='mean')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.tag == args.out_tag:
        parser.error('--out-tag must differ from the source tag')
    meta = load_metadata(args.params_dir, args.tag)
    if not meta or not meta.get('bayesian'):
        parser.error('Source must be a native Bayesian ABCD checkpoint')
    directory = Path(args.params_dir)
    with (directory / f'bayesian_{args.tag}.pkl').open('rb') as f:
        checkpoint = pickle.load(f)
    variables = export_world(checkpoint, args.seed, args.mode == 'sample')
    targets = [directory / f'{h.upper()}_{args.out_tag}_params.pkl' for h in variables]
    targets.append(directory / f'{args.out_tag}_metadata.pkl')
    if any(path.exists() for path in targets):
        parser.error('Output tag already exists; choose a new tag')
    for h, value in variables.items():
        with (directory / f'{h.upper()}_{args.out_tag}_params.pkl').open('wb') as f:
            pickle.dump(jax.device_get(value), f)
    meta = deepcopy(meta)
    meta['posterior_world'] = dict(source_tag=args.tag, mode=args.mode, seed=args.seed)
    meta['config']['tag'] = args.out_tag
    save_metadata(directory, args.out_tag, meta)
    print(f'Exported {args.mode} world to tag {args.out_tag}; reuse it for every pitch in that world.')


if __name__ == '__main__':
    main()
