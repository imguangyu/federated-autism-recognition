"""
Visualize MMASD+ skeleton data to verify conversion and SMPL24 mapping.

Modes:
  1. From converted npz: show one sample's skeleton (single frame or animation).
  2. From raw MMASD+ folder: load frame npzs, apply joint map, show skeleton.

Usage:
  cd FreqMixFormer
  # Visualize first sample from npz (one frame + mid + last)
  python data/MMASD+/visualize_skeleton.py --npz data/MMASD+/MMASD+_CS.npz --sample 0 --frames 0,50,100

  # Animate first sample (saves gif or shows interactively)
  python data/MMASD+/visualize_skeleton.py --npz data/MMASD+/MMASD+_CS.npz --sample 0 --animate --out vis_sample0.gif

  # From raw folder (same joint mapping as conversion)
  python data/MMASD+/visualize_skeleton.py --raw /data/MMASD+/Romp_Data_All_Actions/Twist_Pose/Twist_pose/processed_tw_41063_D1_007_y_3_1 --joint-map smpl24
"""

import os
import os.path as osp
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Reuse conversion constants and helpers
try:
    from mmasd_to_freqmixformer_npz import (
        get_joint_indices,
        load_sequence,
        NUM_JOINTS_NTU,
    )
except ImportError:
    import sys
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    from mmasd_to_freqmixformer_npz import (
        get_joint_indices,
        load_sequence,
        NUM_JOINTS_NTU,
    )

# NTU 25-joint skeleton edges (0-based); use only when data is in NTU joint order.
NTU_SKELETON_EDGES = [
    (1, 0), (1, 20), (20, 2), (2, 3),   # head/spine
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (22, 21),   # left arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (24, 23),  # right arm
    (0, 12), (12, 13), (13, 14), (14, 15),   # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),  # right leg
]

# SMPL 24-joint kinematic tree (0-based). ROMP/MMASD+ first 24 joints follow this order.
# Parent->child: 0 pelvis; 1 L_hip, 2 R_hip, 3 spine1; 4 L_knee, 5 R_knee, 6 spine2; ...
SMPL24_SKELETON_EDGES = [
    (0, 1), (0, 2), (0, 3),       # pelvis -> L_hip, R_hip, spine1
    (1, 4), (2, 5), (3, 6),      # -> L_knee, R_knee, spine2
    (4, 7), (5, 8), (6, 9),      # -> L_ankle, R_ankle, spine3
    (7, 10), (8, 11), (9, 12), (9, 13), (9, 14),   # -> feet, neck, L_collar, R_collar
    (12, 15), (13, 16), (14, 17),   # -> head, L_shoulder, R_shoulder
    (16, 18), (17, 19),            # -> L_elbow, R_elbow
    (18, 20), (19, 21),            # -> L_wrist, R_wrist
    (20, 22), (21, 23),            # -> L_hand, R_hand
]


def get_valid_frames(x_one_sample):
    """x_one_sample: (T, 150). Return number of frames with non-zero person0."""
    T = x_one_sample.shape[0]
    p0 = x_one_sample[:, :75].reshape(T, -1)
    valid = (p0.sum(axis=1) != 0)
    return np.where(valid)[0]


def npz_sample_to_joints(x_one_sample, person=0):
    """x_one_sample: (T, 150). Return (T, 25, 3) for the given person."""
    T = x_one_sample.shape[0]
    if person == 0:
        flat = x_one_sample[:, :75]
    else:
        flat = x_one_sample[:, 75:150]
    return flat.reshape(T, NUM_JOINTS_NTU, 3)


def draw_skeleton_3d(ax, joints, edges=None, color='b', title='', use_smpl24=False):
    """joints: (J, 3). Draw 3D skeleton. use_smpl24: use SMPL24 edges (for ROMP/MMASD+ raw)."""
    if edges is None:
        edges = SMPL24_SKELETON_EDGES if use_smpl24 else NTU_SKELETON_EDGES
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=20)
    for (i, j) in edges:
        if i < joints.shape[0] and j < joints.shape[0]:
            ax.plot(
                [joints[i, 0], joints[j, 0]],
                [joints[i, 1], joints[j, 1]],
                [joints[i, 2], joints[j, 2]],
                color=color, linewidth=1.5, alpha=0.8
            )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    # Equal aspect so skeleton doesn't look stretched
    ptp = np.ptp(joints, axis=0)
    m = max(1e-6, ptp.max())
    c = joints.mean(axis=0)
    ax.set_xlim(c[0] - m / 2, c[0] + m / 2)
    ax.set_ylim(c[1] - m / 2, c[1] + m / 2)
    ax.set_zlim(c[2] - m / 2, c[2] + m / 2)


def visualize_from_npz(npz_path, sample_idx=0, frame_indices=None, animate=False, out_path=None):
    """Load one sample from npz and visualize."""
    data = np.load(npz_path)
    x_train = data['x_train']
    y_train = data['y_train']
    if sample_idx >= len(x_train):
        raise ValueError('sample_idx %d >= train size %d' % (sample_idx, len(x_train)))
    x = x_train[sample_idx]   # (T, 150)
    label_onehot = y_train[sample_idx]
    label_idx = int(np.argmax(label_onehot))
    valid = get_valid_frames(x)
    if len(valid) == 0:
        raise ValueError('Sample %d has no valid frames' % sample_idx)
    T_valid = len(valid)
    joints_seq = npz_sample_to_joints(x, person=0)   # (T, 25, 3)

    if animate:
        return _animate_skeleton(joints_seq, valid, label_idx, sample_idx, out_path)

    if frame_indices is None:
        frame_indices = [0, T_valid // 2, T_valid - 1] if T_valid >= 3 else [0]
    else:
        frame_indices = [int(i) for i in frame_indices]

    n_show = len(frame_indices)
    fig, axes = plt.subplots(1, n_show, subplot_kw={'projection': '3d'}, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]
    # NPZ is filled with SMPL24→25 mapping, so use SMPL24 connectivity
    for k, fi in enumerate(frame_indices):
        if fi >= T_valid:
            fi = T_valid - 1
        t = valid[fi]
        draw_skeleton_3d(
            axes[k],
            joints_seq[t],
            title='Sample %d Label %d Frame %d' % (sample_idx, label_idx, t),
            use_smpl24=True,
        )
    plt.suptitle('From NPZ (person 0, SMPL24)', fontsize=12)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print('Saved:', out_path)
    else:
        plt.savefig(osp.join(osp.dirname(npz_path), 'vis_npz_sample%d.png' % sample_idx), dpi=120, bbox_inches='tight')
        print('Saved: vis_npz_sample%d.png' % sample_idx)
    plt.close()


def _animate_skeleton(joints_seq, valid_frames, label_idx, sample_idx, out_path):
    """Create gif of skeleton over time."""
    try:
        import imageio
    except ImportError:
        print('Install imageio for gif: pip install imageio')
        return
    # Limit frames for gif size
    step = max(1, len(valid_frames) // 60)
    indices = valid_frames[::step][:64]
    frames = []
    for t in indices:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        draw_skeleton_3d(ax, joints_seq[t], title='Sample %d Label %d t=%d' % (sample_idx, label_idx, t), use_smpl24=True)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)
    path = out_path or 'vis_animate_sample%d.gif' % sample_idx
    imageio.mimsave(path, frames, fps=10, loop=0)
    print('Saved animation:', path)


def visualize_from_raw(raw_dir, joint_map='smpl24', frame_indices=None, out_path=None):
    """Load one sample from raw MMASD+ folder and visualize (after joint mapping)."""
    joint_indices = get_joint_indices(joint_map)
    seq, T = load_sequence(raw_dir, joint_indices=joint_indices)
    if seq is None or T == 0:
        raise ValueError('No valid frames in %s' % raw_dir)
    joints_seq = seq   # (T, 25, 3)
    if frame_indices is None:
        frame_indices = [0, T // 2, T - 1] if T >= 3 else [0]
    else:
        frame_indices = [min(int(i), T - 1) for i in frame_indices]
    n_show = len(frame_indices)
    fig, axes = plt.subplots(1, n_show, subplot_kw={'projection': '3d'}, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]
    for k, fi in enumerate(frame_indices):
        draw_skeleton_3d(
            axes[k],
            joints_seq[fi],
            title='Raw frame %d (joint_map=%s)' % (fi, joint_map),
            use_smpl24=True,
        )
    plt.suptitle('From raw MMASD+ (%s, SMPL24 skeleton)' % joint_map, fontsize=12)
    plt.tight_layout()
    path = out_path or osp.join(osp.dirname(raw_dir), 'vis_raw_smpl24.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print('Saved:', path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize MMASD+ skeleton for verification')
    parser.add_argument('--npz', type=str, default=None, help='Converted npz path (e.g. data/MMASD+/MMASD+_CS.npz)')
    parser.add_argument('--raw', type=str, default=None, help='Raw sample dir (e.g. .../Twist_pose/processed_xxx)')
    parser.add_argument('--sample', type=int, default=0, help='Sample index when using --npz')
    parser.add_argument('--frames', type=str, default=None, help='Comma-separated frame indices, e.g. 0,50,100')
    parser.add_argument('--joint-map', type=str, default='smpl24', choices=('first25', 'smpl24'), help='For --raw only')
    parser.add_argument('--animate', action='store_true', help='Output gif animation (npz mode)')
    parser.add_argument('--out', type=str, default=None, help='Output image or gif path')
    args = parser.parse_args()

    frame_indices = None
    if args.frames:
        frame_indices = [int(x.strip()) for x in args.frames.split(',')]

    if args.npz:
        if not osp.isfile(args.npz):
            raise SystemExit('Not found: %s' % args.npz)
        visualize_from_npz(
            args.npz,
            sample_idx=args.sample,
            frame_indices=frame_indices,
            animate=args.animate,
            out_path=args.out,
        )
    elif args.raw:
        if not osp.isdir(args.raw):
            raise SystemExit('Not a dir: %s' % args.raw)
        visualize_from_raw(
            args.raw,
            joint_map=args.joint_map,
            frame_indices=frame_indices,
            out_path=args.out,
        )
    else:
        raise SystemExit('Specify either --npz or --raw. See script docstring for examples.')


if __name__ == '__main__':
    main()
