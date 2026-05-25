#!/usr/bin/env python3
"""
Create a CVPR-style figure: 3 rows (one per theme), each row shows a sequence of
skeleton frames from one sample. Optionally add raw RGB frames if available.
Skeletons are reoriented with head on top.

Usage (from FreqMixFormer root):
  python fl_experiments/create_skeleton_sequence_figure.py
  python fl_experiments/create_skeleton_sequence_figure.py --sample 5 --frames 8 --out fig/skeleton_sequence.pdf
  # With raw RGB (Romp_Data_All_Actions or folder with frame_*.jpg / *.mp4 per sample):
  python fl_experiments/create_skeleton_sequence_figure.py --rgb-root /path/to/Romp_Data_All_Actions --out fig/with_rgb.pdf
  # 3D skeleton view with axes:
  python fl_experiments/create_skeleton_sequence_figure.py --3d --out fig/skeleton_3d.pdf
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# CVPR-style: serif font, tight layout
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
})

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DATA_MMASD = PROJ_ROOT / "data" / "MMASD+"
THEMES = ("theme1", "theme2", "theme3")

# Same as mmasd_to_freqmixformer_npz: theme -> action folder names
THEME_ACTIONS = {
    "theme1": ["Arm_swing", "Body_Swing", "Chest_Expansion", "Squat_Pose"],
    "theme2": ["Drumming", "Marcas_Forward_Shaking", "Marcas_Shaking", "Sing_Clap"],
    "theme3": ["Frog_Pose", "Tree_Pose", "Twist_Pose"],
}
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# SMPL24-style skeleton edges (0-based), 25 joints
SKELETON_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12), (9, 13), (9, 14),
    (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (20, 22), (21, 23),
]
PELVIS_IDX, HEAD_IDX = 0, 15
NUM_JOINTS = 25


def load_sample_joints(npz_path: Path, sample_idx: int, person: int = 0):
    """Load one sample; return (T, 25, 3) and valid frame indices."""
    data = np.load(npz_path)
    x = data.get("x_train", data.get("x_test", np.zeros((0, 1, 150))))
    if sample_idx >= len(x):
        return None, np.array([], dtype=int)
    x_one = x[sample_idx]
    T = x_one.shape[0]
    flat = x_one[:, :75] if person == 0 else x_one[:, 75:150]
    joints = flat.reshape(T, NUM_JOINTS, 3)
    valid = np.where((joints.sum(axis=(1, 2)) != 0))[0]
    return joints, valid


def reorient_head_up(joints: np.ndarray) -> np.ndarray:
    """Reorient so head is on top; return (J, 3) with vertical axis as second (y)."""
    j = np.array(joints, dtype=np.float64)
    diff = j[HEAD_IDX] - j[PELVIS_IDX]
    axis = np.argmax(np.abs(diff))
    if diff[axis] < 0:
        j[:, axis] = -j[:, axis]
    if axis == 0:
        out = j[:, [1, 0, 2]]
    elif axis == 2:
        out = j[:, [0, 2, 1]]
    else:
        out = j
    return out.astype(np.float32)


def draw_skeleton_2d(ax, joints_2d, linecolor="k", linewidth=1.2, jointsize=8):
    """Draw skeleton in 2D. joints_2d: (J, 2) with (x, y), y = up."""
    ax.scatter(joints_2d[:, 0], joints_2d[:, 1], s=jointsize, c=linecolor, zorder=2)
    for (i, j) in SKELETON_EDGES:
        if i < joints_2d.shape[0] and j < joints_2d.shape[0]:
            ax.plot(
                [joints_2d[i, 0], joints_2d[j, 0]],
                [joints_2d[i, 1], joints_2d[j, 1]],
                color=linecolor, linewidth=linewidth, zorder=1,
            )


def draw_skeleton_3d(ax, joints_3d, linecolor="k", linewidth=1.2, jointsize=8):
    """Draw skeleton in 3D. joints_3d: (J, 3) with (x, y, z), y = up after reorient. Axis labels hidden; view set for head up."""
    x, y, z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
    ax.scatter(x, y, z, s=jointsize, c=linecolor)
    for (i, j) in SKELETON_EDGES:
        if i < joints_3d.shape[0] and j < joints_3d.shape[0]:
            ax.plot(
                [x[i], x[j]],
                [y[i], y[j]],
                [z[i], z[j]],
                color=linecolor, linewidth=linewidth,
            )
    # Hide axis and tick labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # Equal data limits (cube) so skeleton scale is consistent
    ptp = np.ptp(joints_3d, axis=0)
    m = max(1e-6, ptp.max())
    c = joints_3d.mean(axis=0)
    r = m / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    # Box aspect 1:1:1 (like Plotly aspectmode="data") so skeleton proportions match Streamlit
    ax.set_box_aspect((1, 1, 1))
    # View: head up, feet down (y is body vertical in data; set as display vertical)
    ax.view_init(elev=20, azim=-60, vertical_axis="y")


def _get_action_folders(root: Path) -> List[str]:
    return sorted([n for n in os.listdir(root) if (root / n).is_dir() and not n.endswith(".zip")])


def _list_sample_dirs(action_dir: Path) -> List[Path]:
    out = []
    for name in sorted(os.listdir(action_dir)):
        path = action_dir / name
        if not path.is_dir():
            continue
        try:
            files = os.listdir(path)
        except OSError:
            continue
        if any(f.startswith("frame_") and f.endswith(".npz") for f in files):
            out.append(path)
        else:
            out.extend(_list_sample_dirs(path))
    return out


def get_raw_sample_dir(rgb_root: Path, theme: str, sample_idx: int) -> Optional[Path]:
    """
    Return the raw sample directory corresponding to (theme, sample_idx) in the theme's train split.
    Uses same order and train split as mmasd_to_freqmixformer_npz (seed 42, ratio 0.8).
    """
    actions = _get_action_folders(rgb_root)
    theme_actions = [a for a in THEME_ACTIONS[theme] if a in actions]
    if not theme_actions:
        return None
    samples = []
    for action in theme_actions:
        samples.extend(_list_sample_dirs(rgb_root / action))
    if not samples:
        return None
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(samples))
    n_train = int(len(indices) * TRAIN_RATIO)
    train_indices = indices[:n_train]
    if sample_idx >= len(train_indices):
        return None
    return samples[train_indices[sample_idx]]


def find_rgb_frames(sample_dir: Path, frame_indices: np.ndarray) -> List[Optional[np.ndarray]]:
    """
    Load RGB frames for given frame indices. Tries: frame_N.jpg, frame_N.png, frame_N.jpeg;
    or a single video file (e.g. .mp4) and extract frames by index.
    Returns list of (H,W,3) uint8 or None for missing.
    """
    result = []
    # Try image sequence: look for any frame_*.<ext> in dir
    for ext in (".jpg", ".png", ".jpeg"):
        any_frame = list(sample_dir.glob(f"frame_*{ext}"))
        if not any_frame:
            continue
        try:
            for fi in frame_indices:
                path = sample_dir / f"frame_{int(fi)}{ext}"
                if path.exists():
                    im = plt.imread(str(path))
                    if im.ndim == 2:
                        im = np.stack([im] * 3, axis=-1)
                    elif im.ndim == 3 and im.shape[-1] == 4:
                        im = im[:, :, :3]
                    result.append(im.astype(np.uint8) if im.max() <= 1 else im)
                else:
                    result.append(None)
            return result
        except Exception:
            result = []
            continue
    # Try video
    videos = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.avi")) + list(sample_dir.glob("*.mov"))
    if videos:
        try:
            import cv2
            cap = cv2.VideoCapture(str(videos[0]))
            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result.append(frame)
                else:
                    result.append(None)
            cap.release()
            return result
        except Exception:
            pass
    return [None] * len(frame_indices)


def main():
    parser = argparse.ArgumentParser(description="Create CVPR-style 3-row skeleton sequence figure (optionally with RGB)")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to use for all themes")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames per row")
    parser.add_argument("--out", type=str, default=None, help="Output path (default: fl_experiments/fig/skeleton_sequence.pdf)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for raster export")
    parser.add_argument("--rgb-root", type=str, default=None, help="Path to Romp_Data_All_Actions (or folder with sample dirs containing frame_*.jpg / *.mp4) to show raw RGB")
    parser.add_argument("--3d", dest="view_3d", action="store_true", help="Draw skeleton in 3D with axes (X, Y, Z)")
    args = parser.parse_args()

    use_3d = getattr(args, "view_3d", False)
    out_path = Path(args.out) if args.out else (SCRIPT_DIR / "fig" / ("skeleton_3d.pdf" if use_3d else "skeleton_sequence.pdf"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cols = args.frames
    rgb_root = Path(args.rgb_root) if args.rgb_root else None

    # Load one sample per theme and get frame indices
    theme_data = {}
    for theme in THEMES:
        npz_path = DATA_MMASD / f"MMASD+_{theme}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ not found: {npz_path}")
        joints_seq, valid = load_sample_joints(npz_path, args.sample, person=0)
        if joints_seq is None or len(valid) == 0:
            raise ValueError(f"Theme {theme}: no valid frames for sample {args.sample}")
        if len(valid) >= n_cols:
            frame_indices = valid[np.linspace(0, len(valid) - 1, n_cols, dtype=int)]
        else:
            frame_indices = np.array(list(valid) + [valid[-1]] * (n_cols - len(valid)), dtype=int)
        theme_data[theme] = (joints_seq, frame_indices)

    # Try load RGB for each theme if --rgb-root set
    theme_rgb: dict = {}
    if rgb_root and rgb_root.is_dir():
        for theme in THEMES:
            raw_dir = get_raw_sample_dir(rgb_root, theme, args.sample)
            if raw_dir is None:
                theme_rgb[theme] = []
                continue
            _, frame_indices = theme_data[theme]
            frames = find_rgb_frames(raw_dir, frame_indices)
            theme_rgb[theme] = frames if any(f is not None for f in frames) else []

    use_rgb = any(len(theme_rgb.get(t, [])) > 0 for t in THEMES)
    if use_rgb:
        n_rows = 2 * len(THEMES)  # RGB row + skeleton row per theme
    else:
        n_rows = len(THEMES)
    if rgb_root and not use_rgb:
        print("Note: --rgb-root given but no RGB frames found (look for frame_*.jpg/.png or *.mp4 in each sample dir).")

    # So 3D box (set_box_aspect 1:1:1) is limited by row height, not width: need cell width >= cell height
    fig_h = 5.0 * n_rows
    fig_w = max(9.0, fig_h * n_cols / n_rows)
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Use GridSpec so row spacing (hspace) is tiny — most height goes to subplots
    gs = GridSpec(
        n_rows, n_cols, figure=fig,
        left=0.06, right=0.98, top=0.96, bottom=0.04,
        hspace=0.04, wspace=0.08,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            if use_3d and (not use_rgb or r % 2 == 1):
                axes[r, c] = fig.add_subplot(gs[r, c], projection="3d")
            else:
                axes[r, c] = fig.add_subplot(gs[r, c])

    for row, theme in enumerate(THEMES):
        joints_seq, frame_indices = theme_data[theme]
        if use_rgb:
            ax_row_rgb = 2 * row
            ax_row_skel = 2 * row + 1
            # RGB row
            rgb_frames = theme_rgb.get(theme, [])
            for col in range(n_cols):
                ax = axes[ax_row_rgb, col]
                ax.axis("off")
                if col < len(rgb_frames) and rgb_frames[col] is not None:
                    img = rgb_frames[col]
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    ax.imshow(img)
                else:
                    ax.set_facecolor("0.9")
                    ax.text(0.5, 0.5, "No RGB", transform=ax.transAxes, ha="center", va="center", fontsize=8)
            axes[ax_row_rgb, 0].text(-0.2, 0.5, theme.replace("theme", "Theme ") + " (RGB)", transform=axes[ax_row_rgb, 0].transAxes, fontsize=14, fontweight="bold", va="center", ha="right")
            # Skeleton row
            skel_row = ax_row_skel
        else:
            skel_row = row

        # Skeleton row
        for col, t in enumerate(frame_indices):
            ax = axes[skel_row, col]
            joints = joints_seq[t]
            joints_up = reorient_head_up(joints)
            if use_3d:
                draw_skeleton_3d(ax, joints_up, linecolor="k", linewidth=1.2, jointsize=6)
            else:
                xy = joints_up[:, [0, 1]]
                draw_skeleton_2d(ax, xy, linecolor="k", linewidth=1.2, jointsize=6)
                ax.set_aspect("equal")
                ax.axis("off")
                margin = max(np.ptp(xy) * 0.15, 1e-6)
                ax.set_xlim(xy[:, 0].min() - margin, xy[:, 0].max() + margin)
                ax.set_ylim(xy[:, 1].min() - margin, xy[:, 1].max() + margin)
                ax.invert_xaxis()
        label_s = theme.replace("theme", "Theme ") + (" (Skeleton)" if use_rgb else "")
        label_x = -0.35 if not use_rgb else -0.2
        ax0 = axes[skel_row, 0]
        if use_3d:
            ax0.text2D(label_x, 0.5, label_s, transform=ax0.transAxes, fontsize=14, fontweight="bold", va="center", ha="right")
        else:
            ax0.text(label_x, 0.5, label_s, transform=ax0.transAxes, fontsize=14, fontweight="bold", va="center", ha="right")

    fig.savefig(out_path, bbox_inches="tight", dpi=args.dpi)
    print(f"Saved: {out_path}")
    if out_path.suffix.lower() == ".pdf":
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight", dpi=args.dpi)
        print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
