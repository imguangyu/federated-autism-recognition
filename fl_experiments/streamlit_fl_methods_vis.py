#!/usr/bin/env python3
"""
Streamlit app to visualize per-round accuracies for different FL methods on MMASD+.

Run from FreqMixFormer root:

    streamlit run fl_experiments/streamlit_fl_methods_vis.py
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent  # FreqMixFormer
WORKSPACES_ROOT = SCRIPT_DIR / "workspaces"
LOG_ROOT = SCRIPT_DIR / "logs"
DATA_MMASD = PROJ_ROOT / "data" / "MMASD+"
THEMES = ("theme1", "theme2", "theme3")
METHODS = ("fedavg", "fedprox", "fedbn", "fedper", "local", "apfl")
APFL_EARLIEST_LABEL = "apfl_earliest"

# SMPL24-style skeleton edges (0-based) for MMASD+ converted npz (25 joints)
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
# Joint 0 = pelvis, 15 = head (for head-on-top reorientation)
PELVIS_IDX, HEAD_IDX = 0, 15
NUM_JOINTS = 25


def find_runs_from_logs() -> Dict[str, Dict[str, Path]]:
    """Parse fl_*.log to find (run_name -> {method, workspace})."""
    runs: Dict[str, Dict[str, Path]] = {}
    if not LOG_ROOT.exists():
        return runs
    for log_path in LOG_ROOT.glob("fl_*.log"):
        name = log_path.stem  # e.g. fl_fedavg_warmup
        if not name.startswith("fl_"):
            continue
        run_name = name[3:]  # e.g. fedavg_warmup
        method = run_name.split("_")[0]  # base method name
        workspace = None
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if "Using NVFLARE workspace:" in line:
                        workspace = line.split("Using NVFLARE workspace:")[-1].strip()
                        break
        except Exception:
            continue
        if not workspace or not os.path.isdir(workspace):
            continue
        runs[run_name] = {"method": method, "workspace": Path(workspace)}
    return runs


def find_latest_workspaces() -> Dict[str, Path]:
    """Fallback: latest workspace dir per method when no logs are available."""
    out: Dict[str, Path] = {}
    if not WORKSPACES_ROOT.exists():
        return out
    for m in METHODS:
        prefix = f"nvflare_workspace_{m}_"
        candidates = [d for d in os.listdir(WORKSPACES_ROOT) if d.startswith(prefix)]
        if not candidates:
            continue
        candidates.sort(reverse=True)
        out[m] = WORKSPACES_ROOT / candidates[0]
    return out


def find_apfl_workspaces() -> List[tuple]:
    """Find APFL workspaces; returns [(run_label, workspace_path), ...] with earliest first.
    First entry is (APFL_EARLIEST_LABEL, path) for the earliest run, then (dirname, path) for others.
    """
    if not WORKSPACES_ROOT.exists():
        return []
    prefix = "nvflare_workspace_apfl_"
    candidates = [
        d for d in os.listdir(WORKSPACES_ROOT)
        if d.startswith(prefix) and (WORKSPACES_ROOT / d / "server" / "simulate_job" / "tb_events").exists()
    ]
    if not candidates:
        return []
    candidates.sort()  # earliest by name (timestamp in name)
    out: List[tuple] = [(APFL_EARLIEST_LABEL, WORKSPACES_ROOT / candidates[0])]
    for d in candidates[1:]:
        out.append((d, WORKSPACES_ROOT / d))
    return out


def load_per_class_from_workspace(ws_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load per-class accuracy CSV from FL workspace (client writes per_class_acc.csv in app_themeX).
    Returns dict theme -> DataFrame with columns class_idx, accuracy."""
    out: Dict[str, pd.DataFrame] = {}
    for theme in THEMES:
        csv_path = ws_dir / theme / "simulate_job" / f"app_{theme}" / "per_class_acc.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            # Last row is latest round; columns class_0, class_1, ...
            class_cols = [c for c in df.columns if c.startswith("class_")]
            if not class_cols:
                continue
            row = df.iloc[-1]
            accs = [float(row[c]) for c in class_cols]
            out[theme] = pd.DataFrame({"class_idx": range(len(accs)), "accuracy": accs})
        except Exception:
            continue
    return out


def load_method_rounds(method: str, ws_dir: Path) -> pd.DataFrame:
    """Load per-round accuracy from TensorBoard events for one method."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        st.error("tensorboard is not installed. Please run `pip install tensorboard` in your env.")
        return pd.DataFrame()

    tb_root = ws_dir / "server" / "simulate_job" / "tb_events"
    if not tb_root.exists():
        return pd.DataFrame()

    rows: List[Dict] = []
    for theme in THEMES:
        theme_dir = tb_root / theme
        if not theme_dir.exists():
            continue
        event_files = [f for f in theme_dir.iterdir() if f.name.startswith("events.")]
        if not event_files:
            continue
        # Use the first event file (there is typically only one per run)
        ea = event_accumulator.EventAccumulator(str(event_files[0]))
        try:
            ea.Reload()
        except Exception:
            continue
        tags = ea.Tags().get("scalars", [])
        if not tags:
            continue
        # Prefer accuracy/{theme}, otherwise any tag containing "accuracy"
        tag = None
        preferred = f"accuracy/{theme}"
        if preferred in tags:
            tag = preferred
        else:
            for t in tags:
                if "accuracy" in t:
                    tag = t
                    break
        if tag is None:
            continue
        scalars = ea.Scalars(tag)
        for s in scalars:
            rows.append(
                {
                    "method": method,
                    "theme": theme,
                    "round": s.step,
                    "accuracy": s.value * 100.0,
                }
            )

    return pd.DataFrame(rows)


def load_skeleton_from_npz(npz_path: Path, sample_idx: int, person: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Load one sample from npz. Returns (joints_seq (T, 25, 3), label_idx, num_frames) or None."""
    if not npz_path.exists():
        return None
    try:
        data = np.load(npz_path)
        x = data["x_train"] if "x_train" in data else data["x_test"]
        y = data["y_train"] if "y_train" in data else data["y_test"]
    except Exception:
        return None
    if sample_idx >= len(x):
        return None
    x_one = x[sample_idx]  # (T, 150)
    T = x_one.shape[0]
    if person == 0:
        flat = x_one[:, :75]
    else:
        flat = x_one[:, 75:150]
    joints_seq = flat.reshape(T, NUM_JOINTS, 3)
    label_onehot = y[sample_idx]
    label_idx = int(np.argmax(label_onehot))
    valid = np.where((joints_seq.sum(axis=(1, 2)) != 0))[0]
    return joints_seq, label_idx, len(valid)


def reorient_skeleton_head_up(joints: np.ndarray) -> np.ndarray:
    """Reorient so head is on top and feet on bottom. joints: (J, 3). Returns (J, 3) with y as vertical up."""
    j = np.array(joints, dtype=np.float64)
    head = j[HEAD_IDX]
    pelvis = j[PELVIS_IDX]
    diff = head - pelvis
    # Which axis has largest head-pelvis separation -> use as vertical
    axis = np.argmax(np.abs(diff))
    vertical = diff[axis]
    if vertical < 0:
        j[:, axis] = -j[:, axis]
    # Permute so vertical axis becomes y for plot (Plotly 3D has y up by default with camera)
    axes_order = [0, 1, 2]
    if axis == 0:
        axes_order = [1, 0, 2]  # display (y,x,z) -> vertical in y
    elif axis == 2:
        axes_order = [0, 2, 1]  # display (x,z,y) -> vertical in y
    out = j[:, axes_order]
    return out.astype(np.float32)


def plot_skeleton_3d(joints: np.ndarray, title: str = "") -> go.Figure:
    """Plot 3D skeleton with Plotly. joints: (J, 3) after reorient (y = up)."""
    x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=5, color="blue"), name="Joints")
    )
    for (i, j) in SKELETON_EDGES:
        if i < len(x) and j < len(x):
            fig.add_trace(
                go.Scatter3d(
                    x=[x[i], x[j]],
                    y=[y[i], y[j]],
                    z=[z[i], z[j]],
                    mode="lines",
                    line=dict(color="darkblue", width=3),
                    showlegend=False,
                )
            )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y (up)",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0)),
        ),
        height=500,
    )
    return fig


def load_alpha_rounds(run_name: str, ws_dir: Path) -> pd.DataFrame:
    """Load per-round alpha from TensorBoard events (APFL). Returns empty if no alpha scalars."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return pd.DataFrame()

    tb_root = ws_dir / "server" / "simulate_job" / "tb_events"
    if not tb_root.exists():
        return pd.DataFrame()

    rows: List[Dict] = []
    for theme in THEMES:
        theme_dir = tb_root / theme
        if not theme_dir.exists():
            continue
        event_files = [f for f in theme_dir.iterdir() if f.name.startswith("events.")]
        if not event_files:
            continue
        ea = event_accumulator.EventAccumulator(str(event_files[0]))
        try:
            ea.Reload()
        except Exception:
            continue
        tags = ea.Tags().get("scalars", [])
        tag = f"alpha/{theme}"
        if tag not in tags:
            continue
        scalars = ea.Scalars(tag)
        for s in scalars:
            rows.append(
                {
                    "run": run_name,
                    "theme": theme,
                    "round": s.step,
                    "alpha": s.value,
                }
            )
    return pd.DataFrame(rows)


def render_dataset_tab():
    """Dataset / Skeleton inspection tab: load npz, reorient head-up, plot 3D."""
    st.subheader("Inspect skeleton data (MMASD+)")
    st.caption("Load a sample from the converted npz; skeleton is reoriented so head is on top and feet on bottom (Y = up).")
    theme = st.selectbox("Theme", options=list(THEMES), key="skel_theme")
    npz_path = DATA_MMASD / f"MMASD+_{theme}.npz"
    if not npz_path.exists():
        st.warning(f"NPZ not found: {npz_path}. Put converted MMASD+ npz files in data/MMASD+/.")
        return
    try:
        data = np.load(npz_path)
        n_samples = len(data.get("x_train", data.get("x_test", [])))
    except Exception:
        st.error(f"Failed to load {npz_path}")
        return
    if n_samples == 0:
        st.warning("No samples in this npz.")
        return
    sample_idx = st.number_input("Sample index", min_value=0, max_value=max(0, n_samples - 1), value=0, step=1, key="skel_sample")
    result = load_skeleton_from_npz(npz_path, sample_idx, person=0)
    if result is None:
        st.warning("Could not load this sample.")
        return
    joints_seq, label_idx, num_valid = result
    frame_max = max(0, num_valid - 1)
    frame_idx = st.slider("Frame", min_value=0, max_value=frame_max, value=frame_max // 2, key="skel_frame")
    if num_valid == 0:
        st.warning("Sample has no valid frames.")
        return
    # Map slider frame to valid frame index
    valid = np.where((joints_seq.sum(axis=(1, 2)) != 0))[0]
    if len(valid) == 0:
        st.warning("No valid frames.")
        return
    frame_idx = min(frame_idx, len(valid) - 1)
    t = valid[frame_idx]
    joints = joints_seq[t]
    joints_up = reorient_skeleton_head_up(joints)
    fig = plot_skeleton_3d(
        joints_up,
        title=f"Sample {sample_idx} · Label {label_idx} · Frame {t} (theme={theme})",
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Raw joint coordinates (first frame, after reorient)"):
        st.dataframe(pd.DataFrame(joints_up, columns=["x", "y", "z"]))


def main():
    st.set_page_config(page_title="MMASD+ FL methods", layout="wide")
    st.title("MMASD+ FL methods")

    tab_fl, tab_data = st.tabs(["FL accuracy", "Dataset / Skeleton"])

    with tab_fl:
        st.subheader("Per-round accuracy")
        runs = find_runs_from_logs()
    # Merge in APFL workspaces (earliest + others) so they appear in run list
    for label, path in find_apfl_workspaces():
        runs[label] = {"method": "apfl", "workspace": path}
    use_runs = bool(runs)

    if use_runs:
        st.sidebar.markdown("### Runs (from logs + APFL workspaces)")
        available_runs = sorted(runs.keys())
        selected_runs = st.sidebar.multiselect(
            "Select FL runs (apfl_earliest = earliest APFL)",
            options=available_runs,
            default=available_runs,
        )
        if not selected_runs:
            st.info("Please select at least one run.")
            run_to_workspace = {}
        else:
            run_to_workspace = {name: runs[name]["workspace"] for name in selected_runs}
    else:
        latest_ws = find_latest_workspaces()
        method_to_ws = dict(latest_ws)
        for label, path in find_apfl_workspaces():
            method_to_ws[label] = path
        if not method_to_ws:
            st.error(f"No FL workspaces found under {WORKSPACES_ROOT}")
            st.stop()
        st.sidebar.markdown("### Methods")
        available_methods = sorted(method_to_ws.keys())
        selected_runs = st.sidebar.multiselect(
            "Select FL methods (apfl_earliest = earliest APFL)",
            options=available_methods,
            default=available_methods,
        )
        run_to_workspace = {m: method_to_ws[m] for m in selected_runs}

    st.sidebar.markdown("### Themes")
    selected_themes = st.sidebar.multiselect(
        "Select themes",
        options=list(THEMES),
        default=list(THEMES),
    )

    st.sidebar.markdown("### Round range (filter)")
    round_min = st.sidebar.number_input("Min round", min_value=0, value=0, step=1)
    round_max = st.sidebar.number_input("Max round", min_value=0, value=999, step=1)

    if not selected_runs or not selected_themes:
        st.info("Please select at least one run and one theme.")
    else:
        # Load data for selected runs / methods
        dfs = []
        for run_name in selected_runs:
            ws = run_to_workspace.get(run_name)
            if ws is None:
                continue
            df_m = load_method_rounds(run_name, ws)
            if not df_m.empty:
                dfs.append(df_m)

        if not dfs:
            st.warning("No accuracy data found in TensorBoard events. Make sure FL runs have finished and tensorboard is installed.")
        else:
            data = pd.concat(dfs, ignore_index=True)
            data = data[data["theme"].isin(selected_themes)]
            data = data[(data["round"] >= round_min) & (data["round"] <= round_max)]

            st.subheader("Per-round accuracy (global view)")
            fig = px.line(
                data,
                x="round",
                y="accuracy",
                color="method",
                facet_col="theme",
                facet_col_wrap=3,
                title="Per-round accuracy by method and theme",
            )
            fig.update_yaxes(title_text="Accuracy (%)")
            fig.update_xaxes(title_text="Round")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

            st.sidebar.markdown("---")
            st.sidebar.markdown("### Export CSV")
            acc_csv = data.to_csv(index=False)
            st.sidebar.download_button(
                "Export per-round accuracy (filtered)",
                data=acc_csv,
                file_name="fl_per_round_accuracy.csv",
                mime="text/csv",
                key="export_acc_csv",
            )

            st.subheader("Final accuracy summary (last round)")
            final_rows = []
            for (m, t), group in data.groupby(["method", "theme"]):
                last = group.sort_values("round").iloc[-1]
                final_rows.append(
                    {
                        "method": m,
                        "theme": t,
                        "round": int(last["round"]),
                        "accuracy (%)": round(float(last["accuracy"]), 2),
                    }
                )
            if final_rows:
                final_df = pd.DataFrame(final_rows).sort_values(["method", "theme"])
                wide = final_df.pivot(index="method", columns="theme", values="accuracy (%)")
                wide = wide.reindex(columns=selected_themes)
                if selected_themes:
                    wide["Avg (%)"] = wide[selected_themes].mean(axis=1)
                st.dataframe(wide, use_container_width=True)
                summary_csv = wide.to_csv()
                st.download_button(
                    "Export final summary as CSV",
                    data=summary_csv,
                    file_name="fl_final_summary.csv",
                    mime="text/csv",
                    key="export_summary_csv",
                )

            # Per-class accuracy (latest round, from client CSV)
            st.subheader("Per-class accuracy (latest round)")
            run_for_class = st.selectbox(
                "Run for per-class",
                options=selected_runs,
                key="per_class_run",
            )
            ws_for_class = run_to_workspace.get(run_for_class) if run_for_class else None
            if ws_for_class is not None:
                per_class_by_theme = load_per_class_from_workspace(ws_for_class)
                if per_class_by_theme:
                    theme_options = [t for t in selected_themes if t in per_class_by_theme]
                    theme_for_class = st.selectbox(
                        "Theme",
                        options=theme_options or list(per_class_by_theme.keys()),
                        key="per_class_theme",
                    )
                    if theme_for_class and theme_for_class in per_class_by_theme:
                        pc_df = per_class_by_theme[theme_for_class]
                        fig_pc = px.bar(
                            pc_df,
                            x="class_idx",
                            y="accuracy",
                            title=f"{run_for_class} — {theme_for_class} test accuracy per class (latest round)",
                            labels={"class_idx": "Class", "accuracy": "Accuracy"},
                        )
                        fig_pc.update_yaxes(tickformat=".0%")
                        st.plotly_chart(fig_pc, use_container_width=True)
                    else:
                        st.caption("No per-class data for selected theme.")
                else:
                    st.caption("No per_class_acc.csv found for this run. Re-run FL with the updated client to generate per-class metrics.")
            else:
                st.caption("Select a run to view per-class accuracy.")

            # Alpha over rounds (APFL runs only)
            alpha_dfs = []
            for run_name in selected_runs:
                ws = run_to_workspace.get(run_name)
                if ws is None:
                    continue
                df_alpha = load_alpha_rounds(run_name, ws)
                if not df_alpha.empty:
                    alpha_dfs.append(df_alpha)
            if alpha_dfs:
                alpha_data = pd.concat(alpha_dfs, ignore_index=True)
                alpha_data = alpha_data[alpha_data["theme"].isin(selected_themes)]
                alpha_data = alpha_data[(alpha_data["round"] >= round_min) & (alpha_data["round"] <= round_max)]
                st.subheader("Alpha over rounds (APFL)")
                fig_alpha = px.line(
                    alpha_data,
                    x="round",
                    y="alpha",
                    color="run",
                    facet_col="theme",
                    facet_col_wrap=3,
                    title="APFL alpha by run and theme",
                )
                fig_alpha.update_yaxes(title_text="Alpha")
                fig_alpha.update_xaxes(title_text="Round")
                fig_alpha.update_layout(height=400)
                st.plotly_chart(fig_alpha, use_container_width=True)
                alpha_csv = alpha_data.to_csv(index=False)
                st.download_button(
                    "Export alpha (APFL) as CSV",
                    data=alpha_csv,
                    file_name="apfl_alpha.csv",
                    mime="text/csv",
                    key="export_alpha_csv",
                )

    with tab_data:
        render_dataset_tab()


if __name__ == "__main__":
    main()

