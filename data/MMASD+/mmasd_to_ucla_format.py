"""
Convert MMASD+ to NW-UCLA format: one JSON per sample with skeletons (T, 20, 3) and label.

Output:
  - data/MMASD+/all_sqe/<file_name>.json for each sample
  - data/MMASD+/train_data_dict.json, val_data_dict.json (list of {file_name, length, label})

Then use feeders.feeder_mmasd.Feeder with graph.ucla.Graph (20 joints, 1 person).

Usage:
  cd FreqMixFormer
  python data/MMASD+/mmasd_to_ucla_format.py --root /data/MMASD+/Romp_Data_All_Actions --out-dir data/MMASD+
  python data/MMASD+/mmasd_to_ucla_format.py --joint-map smpl24 --out-dir data/MMASD+

  # Per-theme subsets (see data/MMASD+/README.md):
  python data/MMASD+/mmasd_to_ucla_format.py --subset theme1 --out-dir data/MMASD+   # -> data/MMASD+/theme1/
  python data/MMASD+/mmasd_to_ucla_format.py --subset theme2 --out-dir data/MMASD+   # -> data/MMASD+/theme2/
  python data/MMASD+/mmasd_to_ucla_format.py --subset theme3 --out-dir data/MMASD+   # -> data/MMASD+/theme3/
"""

import os
import os.path as osp
import argparse
import json
import numpy as np
import re

try:
    from mmasd_to_freqmixformer_npz import (
        get_action_folders,
        list_sample_dirs,
        get_joint_indices,
        load_sequence,
        THEME_ACTIONS,
    )
except ImportError:
    import sys
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    from mmasd_to_freqmixformer_npz import (
        get_action_folders,
        list_sample_dirs,
        get_joint_indices,
        load_sequence,
        THEME_ACTIONS,
    )

NUM_JOINTS_UCLA = 20  # NW-UCLA uses 20 joints
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/guangyusun/workspace/flar/data/MMASD+/Romp_Data_All_Actions')
    parser.add_argument('--out-dir', type=str, default='data/MMASD+',
                        help='Output dir: all_sqe/ and *_data_dict.json will be created here')
    parser.add_argument('--joint-map', type=str, default='smpl24', choices=('first25', 'smpl24'))
    parser.add_argument('--subset', type=str, default='all', choices=('all', 'theme1', 'theme2', 'theme3'),
                        help='Subset by theme: theme1 (4 classes), theme2 (4), theme3 (3), or all (11). See README.md.')
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    all_actions = get_action_folders(args.root)
    if not all_actions:
        raise SystemExit('No action folders under %s' % args.root)
    # Global label 0..10 for all subsets so num_class is always 11
    num_class = len(all_actions)  # 11
    action_to_global_label = {a: i for i, a in enumerate(all_actions)}
    if args.subset == 'all':
        actions = all_actions
        out_subdir = ''
    else:
        theme_list = THEME_ACTIONS[args.subset]
        actions = [a for a in theme_list if a in all_actions]
        if len(actions) != len(theme_list):
            missing = set(theme_list) - set(all_actions)
            raise SystemExit('Subset %s: missing folders: %s' % (args.subset, missing))
        out_subdir = args.subset  # e.g. theme1 -> output under out_dir/theme1/
    joint_indices_25 = get_joint_indices(args.joint_map)  # 25 indices into 71

    # Collect (sample_dir, action_name, global_label_0based 0..10)
    samples = []
    for action in actions:
        action_dir = osp.join(args.root, action)
        for sample_dir in list_sample_dirs(action_dir):
            samples.append((sample_dir, action, action_to_global_label[action]))

    np.random.seed(args.seed)
    perm = np.random.permutation(len(samples))
    n_train = int(len(samples) * args.train_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    out_dir = osp.abspath(args.out_dir)
    if out_subdir:
        out_dir = osp.join(out_dir, out_subdir)
    all_sqe = osp.join(out_dir, 'all_sqe')
    os.makedirs(all_sqe, exist_ok=True)

    def process_split(indices, split_name):
        data_dict = []
        for i, idx in enumerate(indices):
            sample_dir, action, label_0 = samples[idx]
            seq, T = load_sequence(sample_dir, joint_indices=joint_indices_25)
            if seq is None or T == 0:
                continue
            # 25 -> 20: take first 20 joints (same as NTU->UCLA simple mapping)
            seq_20 = seq[:, :NUM_JOINTS_UCLA, :].astype(np.float64)  # (T, 20, 3)
            T = seq_20.shape[0]
            file_name = 'mmasd_%s_%05d' % (action.replace(' ', '_'), idx)
            obj = {
                'file_name': file_name,
                'skeletons': seq_20.tolist(),  # list of list of [x,y,z]
                'label': label_0 + 1,  # 1-based 1..11 (global), like NW-UCLA
            }
            json_path = osp.join(all_sqe, file_name + '.json')
            with open(json_path, 'w') as f:
                json.dump(obj, f, separators=(',', ':'))
            data_dict.append({'file_name': file_name, 'length': T, 'label': label_0 + 1})  # 1-based 1..11
        return data_dict

    train_data_dict = process_split(train_idx, 'train')
    val_data_dict = process_split(val_idx, 'val')

    train_path = osp.join(out_dir, 'train_data_dict.json')
    val_path = osp.join(out_dir, 'val_data_dict.json')
    with open(train_path, 'w') as f:
        json.dump(train_data_dict, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_data_dict, f, indent=2)

    print('Saved NW-UCLA format under %s' % out_dir)
    print('  Subset: %s | all_sqe/: %d JSONs (train+val)' % (args.subset, len(train_data_dict) + len(val_data_dict)))
    print('  train_data_dict.json: %d samples' % len(train_data_dict))
    print('  val_data_dict.json: %d samples' % len(val_data_dict))
    print('  num_class: %d, joints: %d (UCLA style)' % (num_class, NUM_JOINTS_UCLA))


if __name__ == '__main__':
    main()
