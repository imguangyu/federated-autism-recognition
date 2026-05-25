"""
Visualize original MMASD+ frame data: load each frame_*.npz and show 3D pose (joints)
and 2D projection (pj2d_org) to verify correctness.

No conversion — uses raw ROMP output: joints (1,71,3), pj2d_org (1,71,2).
Uses first 24 joints (SMPL) for skeleton drawing.

Usage (from FreqMixFormer):
  python data/MMASD+/visualize_original_frames_pose.py --sample-dir /path/to/processed_xxx
  python data/MMASD+/visualize_original_frames_pose.py --sample-dir /path/to/processed_xxx --frames 0,30,60,90 --out vis_original.png

  # Export skeleton motion as GIF
  python data/MMASD+/visualize_original_frames_pose.py --sample-dir /path/to/processed_xxx --gif skeleton_motion.gif --fps 15 --max-frames 80
"""

import os
import os.path as osp
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# SMPL 24-joint kinematic tree (0-based)
SMPL24_SKELETON_EDGES = [
    (0, 1), (0, 2), (0, 3),       # pelvis -> L_hip, R_hip, spine1
    (1, 4), (2, 5), (3, 6),      # -> L_knee, R_knee, spine2
    (4, 7), (5, 8), (6, 9),      # -> L_ankle, R_ankle, spine3
    (7, 10), (8, 11), (9, 12), (9, 13), (9, 14),   # -> feet, neck, collars
    (12, 15), (13, 16), (14, 17),   # -> head, shoulders
    (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23),
]

NUM_SMPL = 24


def list_frame_npz(sample_dir):
    """Return sorted list of frame_*.npz paths and frame indices."""
    files = [f for f in os.listdir(sample_dir) if f.startswith('frame_') and f.endswith('.npz')]
    def key(f):
        m = re.match(r'frame_(\d+)\.npz', f)
        return int(m.group(1)) if m else -1
    files.sort(key=key)
    return [osp.join(sample_dir, f) for f in files], [key(f) for f in files]


def load_frame_npz(path):
    """Load one frame npz. Returns joints_24 (24,3), pj2d_24 (24,2)."""
    with np.load(path, allow_pickle=True) as data:
        joints = data['joints'][0]   # (71, 3)
        pj2d = data['pj2d_org'][0]  # (71, 2)
    joints_24 = np.asarray(joints[:NUM_SMPL], dtype=np.float32)
    pj2d_24 = np.asarray(pj2d[:NUM_SMPL], dtype=np.float32)
    return joints_24, pj2d_24


def draw_skeleton_3d(ax, joints, color='#1f77b4', title='', xlim=None, ylim=None, zlim=None):
    """joints: (24, 3). Optional xlim,ylim,zlim for fixed view (e.g. for GIF)."""
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=20)
    for (i, j) in SMPL24_SKELETON_EDGES:
        if i < joints.shape[0] and j < joints.shape[0]:
            ax.plot(
                [joints[i, 0], joints[j, 0]],
                [joints[i, 1], joints[j, 1]],
                [joints[i, 2], joints[j, 2]],
                color=color, linewidth=1.5, alpha=0.85
            )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title, fontsize=10)
    if xlim is not None and ylim is not None and zlim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    else:
        ptp = np.ptp(joints, axis=0)
        m = max(1e-6, ptp.max())
        c = joints.mean(axis=0)
        r = m / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)


def draw_skeleton_2d(ax, pj2d, color='#ff7f0e', title=''):
    """pj2d: (24, 2). Image coords: x right, y down. Flip y for display."""
    x, y = pj2d[:, 0], pj2d[:, 1]
    y = -y  # so "up" is up in plot
    ax.scatter(x, y, c=color, s=25)
    for (i, j) in SMPL24_SKELETON_EDGES:
        if i < pj2d.shape[0] and j < pj2d.shape[0]:
            ax.plot([x[i], x[j]], [y[i], y[j]], color=color, linewidth=1.5, alpha=0.85)
    ax.set_xlabel('x (image)')
    ax.set_ylabel('-y (image)')
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.autoscale(True)


def export_gif(sample_dir, out_path, fps=15, max_frames=80):
    """Load all frames (or subsample), compute global 3D bbox, render each frame to GIF."""
    try:
        import imageio
    except ImportError:
        raise SystemExit('Install imageio for GIF: pip install imageio')

    paths, frame_indices = list_frame_npz(sample_dir)
    if not paths:
        raise SystemExit('No frame_*.npz in %s' % sample_dir)

    # Subsample if needed
    if len(paths) > max_frames:
        step = len(paths) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        paths = [paths[i] for i in indices]
        frame_indices = [frame_indices[i] for i in indices]

    # Load all joints and compute global bbox
    all_joints = []
    for path in paths:
        j24, _ = load_frame_npz(path)
        all_joints.append(j24)
    stack = np.concatenate(all_joints, axis=0)
    c = np.mean(stack, axis=0)
    ptp = np.ptp(stack, axis=0)
    m = max(1e-6, ptp.max()) * 0.6
    xlim = (c[0] - m, c[0] + m)
    ylim = (c[1] - m, c[1] + m)
    zlim = (c[2] - m, c[2] + m)

    frames = []
    for k, (path, fi) in enumerate(zip(paths, frame_indices)):
        joints_24, _ = load_frame_npz(path)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        draw_skeleton_3d(ax, joints_24, title='Frame %d' % fi, xlim=xlim, ylim=ylim, zlim=zlim)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = np.asarray(buf).reshape((h, w, 4))[:, :, :3]
        frames.append(img)
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print('Saved GIF: %s (%d frames, %g fps)' % (out_path, len(frames), fps))


def main():
    parser = argparse.ArgumentParser(description='Visualize original MMASD+ frame and pose from .npz')
    parser.add_argument('--sample-dir', type=str, required=True,
                        help='Path to one sample dir (e.g. .../Arm_swing/processed_xxx)')
    parser.add_argument('--frames', type=str, default=None,
                        help='Comma-separated frame indices, e.g. 0,30,60,90. Default: 0, mid, last')
    parser.add_argument('--out', type=str, default=None, help='Output image path')
    parser.add_argument('--gif', type=str, default=None, help='Output GIF path (skeleton motion animation)')
    parser.add_argument('--fps', type=int, default=15, help='GIF frames per second (default: 15)')
    parser.add_argument('--max-frames', type=int, default=80, help='Max frames in GIF; subsample if longer (default: 80)')
    args = parser.parse_args()

    sample_dir = osp.abspath(args.sample_dir)
    if not osp.isdir(sample_dir):
        raise SystemExit('Not a directory: %s' % sample_dir)

    if args.gif:
        export_gif(sample_dir, args.gif, fps=args.fps, max_frames=args.max_frames)
        return

    paths, frame_indices = list_frame_npz(sample_dir)
    if not paths:
        raise SystemExit('No frame_*.npz in %s' % sample_dir)

    if args.frames:
        wanted = [int(x.strip()) for x in args.frames.split(',')]
        # Map to actual available frames
        idx_set = set(frame_indices)
        to_show = []
        for w in wanted:
            if w in idx_set:
                to_show.append(w)
            else:
                # nearest
                best = min(frame_indices, key=lambda i: abs(i - w))
                if best not in [t[0] for t in to_show]:
                    to_show.append(best)
        to_show = sorted(set(to_show))[:9]  # at most 9
    else:
        n = len(frame_indices)
        if n <= 3:
            to_show = frame_indices
        else:
            to_show = [frame_indices[0], frame_indices[n // 2], frame_indices[-1]]

    # Build path index: frame_idx -> path
    frame_to_path = dict(zip(frame_indices, paths))
    rows = []
    for fi in to_show:
        path = frame_to_path.get(fi)
        if path is None:
            continue
        joints_24, pj2d_24 = load_frame_npz(path)
        rows.append((fi, joints_24, pj2d_24))

    if not rows:
        raise SystemExit('No frames to show.')

    n_show = len(rows)
    fig = plt.figure(figsize=(10, 4 * n_show))
    for i, (fi, joints_24, pj2d_24) in enumerate(rows):
        ax3d = fig.add_subplot(n_show, 2, i * 2 + 1, projection='3d')
        draw_skeleton_3d(ax3d, joints_24, title='Frame %d — 3D pose (joints)' % fi)
        ax2d = fig.add_subplot(n_show, 2, i * 2 + 2)
        draw_skeleton_2d(ax2d, pj2d_24, title='Frame %d — 2D projection (pj2d_org)' % fi)

    sample_name = osp.basename(sample_dir.rstrip('/'))
    fig.suptitle('Original MMASD+ — %s (%d frames total)' % (sample_name, len(paths)), fontsize=12)
    plt.tight_layout()

    out_path = args.out or osp.join(sample_dir, 'vis_original_frames_pose.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved: %s' % out_path)
    print('  Frames shown: %s' % [r[0] for r in rows])


if __name__ == '__main__':
    main()
