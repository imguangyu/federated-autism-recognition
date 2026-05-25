"""
Visualize and compare NW-UCLA vs MMASD+ skeleton format side-by-side.

- NW-UCLA: 20 joints, 1 person, one JSON per sample (skeletons T,20,3).
- MMASD+:  25 joints (NTU/SMPL24), 1 person from npz (T,150) or raw folder.

Usage (from FreqMixFormer):
  # Compare using first UCLA sample + MMASD+ from npz
  python data/MMASD+/visualize_mmasd_vs_ucla.py --npz data/MMASD+/MMASD+_CS.npz --out data/MMASD+/compare_ucla_mmasd.png

  # Compare using UCLA + MMASD+ from raw folder
  python data/MMASD+/visualize_mmasd_vs_ucla.py --raw /home/guangyusun/workspace/flar/data/MMASD+/Romp_Data_All_Actions/Arm_swing/processed_xxx --out data/MMASD+/compare_ucla_mmasd.png

  # UCLA only (no MMASD+ path): still draws both topologies with UCLA data for 20 joints and a dummy 25-joint pose
  python data/MMASD+/visualize_mmasd_vs_ucla.py --out data/MMASD+/compare_ucla_mmasd.png
"""

import os
import os.path as osp
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paths relative to script or cwd
FREQMF_ROOT = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '../..'))
NW_UCLA_ALL_SQE = osp.join(FREQMF_ROOT, 'data', 'NW-UCLA', 'all_sqe')

# UCLA: 20 joints, 0-based edges from graph/ucla.py inward
UCLA_SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (2, 4), (2, 8), (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

# SMPL 24-joint kinematic tree (0-based). ROMP/MMASD+ uses this order (25 = 24 + pelvis duplicate).
SMPL24_SKELETON_EDGES = [
    (0, 1), (0, 2), (0, 3),       # pelvis -> L_hip, R_hip, spine1
    (1, 4), (2, 5), (3, 6),      # -> L_knee, R_knee, spine2
    (4, 7), (5, 8), (6, 9),      # -> L_ankle, R_ankle, spine3
    (7, 10), (8, 11), (9, 12), (9, 13), (9, 14),   # -> feet, neck, collars
    (12, 15), (13, 16), (14, 17),   # -> head, shoulders
    (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23),
]


def load_ucla_sample(ucla_sqe_dir, file_stem=None):
    """Load one NW-UCLA sample. Returns (skeletons (T,20,3), label)."""
    if not osp.isdir(ucla_sqe_dir):
        raise FileNotFoundError('NW-UCLA all_sqe not found: %s' % ucla_sqe_dir)
    names = [f[:-5] for f in os.listdir(ucla_sqe_dir) if f.endswith('.json')]
    if not names:
        raise FileNotFoundError('No JSON in %s' % ucla_sqe_dir)
    name = file_stem if file_stem and (file_stem + '.json') in os.listdir(ucla_sqe_dir) else sorted(names)[0]
    path = osp.join(ucla_sqe_dir, name + '.json')
    with open(path, 'r') as f:
        obj = json.load(f)
    skeletons = np.array(obj['skeletons'], dtype=np.float32)  # (T, 20, 3)
    label = int(obj.get('label', 0))
    return skeletons, label, name


def load_mmasd_from_npz(npz_path, sample_idx=0):
    """Load one MMASD+ sample from converted npz. Returns (T, 25, 3) person0."""
    data = np.load(npz_path)
    x = data['x_train'][sample_idx]  # (T, 150)
    T = x.shape[0]
    p0 = x[:, :75].reshape(T, 25, 3)
    valid = (p0.sum(axis=(1, 2)) != 0)
    frames = np.where(valid)[0]
    if len(frames) == 0:
        return None, None
    return p0, frames


def load_mmasd_from_raw(raw_dir):
    """Load one MMASD+ sample from raw folder. Returns (T, 25, 3)."""
    try:
        import sys
        sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
        from mmasd_to_freqmixformer_npz import get_joint_indices, load_sequence
    except ImportError:
        from mmasd_to_freqmixformer_npz import get_joint_indices, load_sequence
    seq, T = load_sequence(raw_dir, joint_indices=get_joint_indices('smpl24'))
    if seq is None or T == 0:
        return None
    return seq, np.arange(T)


def draw_skeleton_3d(ax, joints, edges, color='C0', title='', joint_size=25):
    """Draw 3D skeleton. joints: (J, 3), edges: list of (i,j)."""
    J = joints.shape[0]
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=joint_size)
    for (i, j) in edges:
        if i < J and j < J:
            ax.plot(
                [joints[i, 0], joints[j, 0]],
                [joints[i, 1], joints[j, 1]],
                [joints[i, 2], joints[j, 2]],
                color=color, linewidth=1.8, alpha=0.85
            )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title, fontsize=11)
    ptp = np.ptp(joints, axis=0)
    m = max(1e-6, ptp.max())
    c = joints.mean(axis=0)
    r = m / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def make_dummy_25_from_20(joints_20):
    """Map 20 joints to 25: first 20 copied, 20-24 set to pelvis/neck/head approx for display."""
    out = np.zeros((25, 3), dtype=np.float32)
    out[:20] = joints_20
    # NTU 25: 0=pelvis, 1=spine, 20=neck, 2=head, 21-24 hand; approximate
    out[20] = (joints_20[1] + joints_20[2]) * 0.5   # neck
    out[21] = joints_20[6] + (joints_20[6] - joints_20[5]) * 0.3  # left hand
    out[22] = joints_20[7]
    out[23] = joints_20[10] + (joints_20[10] - joints_20[9]) * 0.3
    out[24] = joints_20[11]
    return out


def main():
    parser = argparse.ArgumentParser(description='Visualize NW-UCLA vs MMASD+ skeleton comparison')
    parser.add_argument('--ucla-dir', type=str, default=NW_UCLA_ALL_SQE, help='NW-UCLA all_sqe directory')
    parser.add_argument('--ucla-file', type=str, default=None, help='UCLA sample name (e.g. a01_s01_e00_v01)')
    parser.add_argument('--npz', type=str, default=None, help='MMASD+ converted npz (e.g. data/MMASD+/MMASD+_CS.npz)')
    parser.add_argument('--raw', type=str, default=None, help='MMASD+ raw sample directory')
    parser.add_argument('--frame', type=int, default=None, help='Frame index (default: middle)')
    parser.add_argument('--out', type=str, default=None, help='Output image path')
    args = parser.parse_args()

    # Load UCLA
    ucla_sqe = args.ucla_dir
    if not osp.isdir(ucla_sqe):
        ucla_sqe = osp.join(os.getcwd(), 'data', 'NW-UCLA', 'all_sqe')
    skeletons_ucla, label_ucla, name_ucla = load_ucla_sample(ucla_sqe, args.ucla_file)
    T_ucla = skeletons_ucla.shape[0]
    frame_idx = args.frame if args.frame is not None else T_ucla // 2
    frame_idx = max(0, min(frame_idx, T_ucla - 1))
    joints_ucla = skeletons_ucla[frame_idx]  # (20, 3)

    # Load MMASD+ (npz or raw)
    joints_mmasd = None
    mmasd_title = 'MMASD+ (25 joints)'
    if args.npz and osp.isfile(args.npz):
        p0, valid_frames = load_mmasd_from_npz(args.npz, 0)
        if p0 is not None and len(valid_frames) > 0:
            fi = min(frame_idx, len(valid_frames) - 1)
            joints_mmasd = p0[valid_frames[fi]]
            mmasd_title = 'MMASD+ npz (25 joints, sample 0)'
    if joints_mmasd is None and args.raw and osp.isdir(args.raw):
        seq, frames = load_mmasd_from_raw(args.raw)
        if seq is not None and len(frames) > 0:
            fi = min(frame_idx, len(frames) - 1)
            joints_mmasd = seq[frames[fi]]
            mmasd_title = 'MMASD+ raw (25 joints, SMPL24)'
    if joints_mmasd is None:
        # Use dummy 25 from UCLA 20 so we still show topology comparison
        joints_mmasd = make_dummy_25_from_20(joints_ucla)
        mmasd_title = 'MMASD+ topology (25 joints, from UCLA pose)'

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    draw_skeleton_3d(
        ax1,
        joints_ucla,
        UCLA_SKELETON_EDGES,
        color='#1f77b4',
        title='NW-UCLA (20 joints, 1 person)',
    )
    draw_skeleton_3d(
        ax2,
        joints_mmasd,
        SMPL24_SKELETON_EDGES,
        color='#ff7f0e',
        title=mmasd_title,
    )

    fig.suptitle('Skeleton format: NW-UCLA vs MMASD+ (frame %d)' % frame_idx, fontsize=13)
    plt.tight_layout()

    out_path = args.out or osp.join(osp.dirname(osp.abspath(__file__)), 'compare_ucla_mmasd.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved: %s' % out_path)
    print('  Left: NW-UCLA %s (T=%d, label=%d), 20 joints.' % (name_ucla, T_ucla, label_ucla))
    print('  Right: %s' % mmasd_title)


if __name__ == '__main__':
    main()
