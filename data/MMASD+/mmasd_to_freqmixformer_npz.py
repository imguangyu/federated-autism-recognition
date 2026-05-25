"""
Convert MMASD+ Romp_Data_All_Actions (per-frame npz with 71 joints) into
a single .npz in FreqMixFormer NTU format: x_train, y_train, x_test, y_test
with shape (N, T, 150) and one-hot labels (N, num_class).

Joint mapping (--joint-map):
  smpl24 (default): Use ROMP's first 24 joints as SMPL 24, then duplicate pelvis -> 25.
  first25: Use ROMP joints 0..24.

Usage:
  cd FreqMixFormer
  python data/MMASD+/mmasd_to_freqmixformer_npz.py --root /data/MMASD+/Romp_Data_All_Actions --out data/MMASD+/MMASD+_CS.npz
  python data/MMASD+/mmasd_to_freqmixformer_npz.py --joint-map first25 --out data/MMASD+/MMASD+_CS.npz
"""

import os
import os.path as osp
import argparse
import numpy as np
import re
from collections import defaultdict

# NTU format uses 25 joints; ROMP outputs 71 (typically: 24 SMPL + 30 extra + 17 H36M).
NUM_JOINTS_NTU = 25
NUM_PERSON = 2
COORD = 3
FRAME_DIM = NUM_PERSON * NUM_JOINTS_NTU * COORD  # 150
MAX_FRAMES = 300  # align length (like NTU); feeder will crop/resize with window_size
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ROMP 71 joints: first 24 are SMPL body joints (standard order).
# SMPL24 -> 25 for NTU: use 24 SMPL joints + duplicate pelvis (index 0) as 25th.
SMPL24_NUM = 24
SMPL24_INDICES_IN_ROMP71 = list(range(SMPL24_NUM))  # 0..23
# Map to 25 slots: SMPL 24 + [pelvis again] so indices into 71 are [0..23, 0]
SMPL24_TO_NTU25_ROMP_INDICES = SMPL24_INDICES_IN_ROMP71 + [0]

# MMASD+ three themes (from official README). Folder names under Romp_Data_All_Actions.
THEME_ACTIONS = {
    'theme1': ['Arm_swing', 'Body_Swing', 'Chest_Expansion', 'Squat_Pose'],   # Robotic-assisted therapy
    'theme2': ['Drumming', 'Marcas_Forward_Shaking', 'Marcas_Shaking', 'Sing_Clap'],  # Rhythm
    'theme3': ['Frog_Pose', 'Tree_Pose', 'Twist_Pose'],   # Yoga
}


def get_action_folders(root):
    """List action folder names (excluding zip and files)."""
    actions = []
    for name in sorted(os.listdir(root)):
        path = osp.join(root, name)
        if not osp.isdir(path):
            continue
        if name.endswith('.zip'):
            continue
        actions.append(name)
    return actions


def list_sample_dirs(action_dir):
    """List sample dirs that contain frame_*.npz (recursive: some actions have action/subaction/processed_xxx/)."""
    samples = []
    for name in sorted(os.listdir(action_dir)):
        path = osp.join(action_dir, name)
        if not osp.isdir(path):
            continue
        try:
            files = os.listdir(path)
        except OSError:
            continue
        has_npz = any(f.startswith('frame_') and f.endswith('.npz') for f in files)
        if has_npz:
            samples.append(path)
        else:
            # nested: e.g. Drumming/Drumming_Pose/processed_xxx/
            samples.extend(list_sample_dirs(path))
    return samples


def get_joint_indices(joint_map):
    """
    Return list of 25 indices into ROMP 71 joints.
    joint_map: 'first25' -> [0..24]; 'smpl24' -> SMPL 24 + duplicate pelvis -> 25.
    """
    if joint_map == 'smpl24':
        return SMPL24_TO_NTU25_ROMP_INDICES
    if joint_map == 'first25':
        return list(range(min(NUM_JOINTS_NTU, 71)))
    raise ValueError("joint_map must be 'first25' or 'smpl24', got %s" % joint_map)


def load_sequence(sample_dir, joint_indices=None):
    """
    Load one sample: all frame_*.npz in order -> (T, 71, 3).
    joint_indices: list of 25 indices into 71 (from get_joint_indices).
    Returns (T, 25, 3) and valid frame count.
    """
    files = [f for f in os.listdir(sample_dir) if f.startswith('frame_') and f.endswith('.npz')]
    if not files:
        return None, 0

    def frame_num(f):
        m = re.match(r'frame_(\d+)\.npz', f)
        return int(m.group(1)) if m else -1

    files.sort(key=frame_num)
    frames = []
    for f in files:
        path = osp.join(sample_dir, f)
        try:
            data = np.load(path)
            joints = data['joints']  # (1, 71, 3)
            frames.append(joints[0])
        except Exception:
            continue
    if not frames:
        return None, 0
    seq = np.stack(frames, axis=0).astype(np.float32)  # (T, 71, 3)
    if joint_indices is None:
        joint_indices = list(range(NUM_JOINTS_NTU))
    seq = seq[:, joint_indices, :]  # (T, 25, 3)
    return seq, seq.shape[0]


def sequence_to_ntu_format(seq, max_frames):
    """
    seq: (T, 25, 3). Single person.
    Return (max_frames, 150): person0 = seq (padded/trimmed), person1 = 0.
    """
    T = seq.shape[0]
    # person0: (T, 25, 3) -> (T, 75); person1: (T, 75) zeros
    p0 = seq.reshape(T, -1)   # (T, 75)
    p1 = np.zeros_like(p0)
    frame_vec = np.concatenate([p0, p1], axis=1)  # (T, 150)
    if T >= max_frames:
        frame_vec = frame_vec[:max_frames]
    else:
        pad = np.zeros((max_frames - T, FRAME_DIM), dtype=np.float32)
        frame_vec = np.concatenate([frame_vec, pad], axis=0)
    return frame_vec


def one_hot(labels, num_class):
    n = len(labels)
    out = np.zeros((n, num_class), dtype=np.float32)
    out[np.arange(n), labels] = 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/guangyusun/workspace/flar/data/MMASD+/Romp_Data_All_Actions',
                        help='Romp_Data_All_Actions root')
    parser.add_argument('--out', type=str, default='MMASD+_CS.npz',
                        help='Output npz path (under cwd or absolute)')
    parser.add_argument('--max-frames', type=int, default=MAX_FRAMES)
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--joint-map', type=str, default='smpl24', choices=('first25', 'smpl24'),
                        help='first25: use ROMP joints 0..24. smpl24: use SMPL 24 (0..23) + pelvis duplicate -> 25.')
    parser.add_argument('--subset', type=str, default='all', choices=('all', 'theme1', 'theme2', 'theme3'),
                        help='Subset by theme: theme1 (4 classes), theme2 (4), theme3 (3), or all (11). See data/MMASD+/README.md.')
    args = parser.parse_args()

    all_actions = get_action_folders(args.root)
    if not all_actions:
        raise SystemExit('No action folders found under %s' % args.root)
    # Global label: same 0..10 for all subsets so num_class is always 11
    num_class = len(all_actions)  # 11
    action_to_global_label = {a: i for i, a in enumerate(all_actions)}
    if args.subset == 'all':
        actions = all_actions
    else:
        theme_list = THEME_ACTIONS[args.subset]
        actions = [a for a in theme_list if a in all_actions]
        if len(actions) != len(theme_list):
            missing = set(theme_list) - set(all_actions)
            raise SystemExit('Subset %s: missing folders under root: %s' % (args.subset, missing))

    # Collect (sample_dir, global_label 0..10)
    samples_with_labels = []
    for action in actions:
        action_dir = osp.join(args.root, action)
        for sample_dir in list_sample_dirs(action_dir):
            samples_with_labels.append((sample_dir, action_to_global_label[action]))

    np.random.seed(args.seed)
    indices = np.random.permutation(len(samples_with_labels))
    n_train = int(len(indices) * args.train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    joint_indices = get_joint_indices(args.joint_map)
    print('Joint map: %s (indices into ROMP 71: %s)' % (args.joint_map, joint_indices))

    def build_split(idx_list):
        x_list = []
        y_list = []
        for idx in idx_list:
            sample_dir, label = samples_with_labels[idx]
            seq, _ = load_sequence(sample_dir, joint_indices=joint_indices)
            if seq is None:
                continue
            frame_vec = sequence_to_ntu_format(seq, args.max_frames)
            x_list.append(frame_vec)
            y_list.append(label)
        if not x_list:
            return None, None
        x = np.stack(x_list, axis=0)
        y = one_hot(np.array(y_list, dtype=np.int64), num_class)
        return x, y

    x_train, y_train = build_split(train_indices)
    x_test, y_test = build_split(test_indices)
    if x_train is None and x_test is None:
        raise SystemExit('No valid samples found.')

    if x_train is None:
        x_train = np.zeros((0, args.max_frames, FRAME_DIM), dtype=np.float32)
        y_train = np.zeros((0, num_class), dtype=np.float32)
    if x_test is None:
        x_test = np.zeros((0, args.max_frames, FRAME_DIM), dtype=np.float32)
        y_test = np.zeros((0, num_class), dtype=np.float32)

    out_path = args.out if osp.isabs(args.out) else osp.join(os.getcwd(), args.out)
    np.savez(out_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print('Saved: %s' % out_path)
    print('Subset: %s | num_class=%d (global 0..10) | actions in subset: %s' % (args.subset, num_class, actions))
    print('Train: %d, Test: %d' % (len(x_train), len(x_test)))
    print('x_train %s, y_train %s' % (x_train.shape, y_train.shape))
    print('x_test %s, y_test %s' % (x_test.shape, y_test.shape))


if __name__ == '__main__':
    main()
