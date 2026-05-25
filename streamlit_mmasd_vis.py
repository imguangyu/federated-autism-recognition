#!/usr/bin/env python3
"""
Streamlit app to visualize MMASD+ theme training and testing (theme1, theme2, theme3).
Run from FreqMixFormer directory: streamlit run streamlit_mmasd_vis.py
"""
import re
import os
import glob
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Default work dir relative to script (FreqMixFormer)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = SCRIPT_DIR / "work_dir" / "mmasd"
THEMES = ["theme1", "theme2", "theme3"]
# Fixed colors per theme so they match across all charts
THEME_COLORS = {"theme1": "#636EFA", "theme2": "#EF553B", "theme3": "#00CC96"}


def _to_float(s: str) -> float:
    """Convert string to float, stripping trailing period (e.g. '7.67.' from log)."""
    return float(s.rstrip("."))


def parse_log_txt(log_path: Path) -> pd.DataFrame:
    """Parse work_dir/mmasd/<theme>/log.txt into a dataframe of epochs and metrics."""
    if not log_path.exists():
        return pd.DataFrame()

    rows = []
    with open(log_path, "r") as f:
        content = f.read()

    # Training epoch: N
    train_epoch_re = re.compile(r"Training epoch: (\d+)")
    # Mean training loss: X. Mean training acc: Y%.
    train_metrics_re = re.compile(
        r"Mean training loss: ([\d.]+)\.\s+Mean training acc: ([\d.]+)%"
    )
    # Eval epoch: N
    # Mean test loss of K batches: X.
    test_loss_re = re.compile(r"Mean test loss of \d+ batches: ([\d.]+)")
    # Top1: X%  Top5: Y%
    top1_re = re.compile(r"Top1: ([\d.]+)%")
    top5_re = re.compile(r"Top5: ([\d.]+)%")

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        m_te = train_epoch_re.search(lines[i])
        if m_te:
            epoch = int(m_te.group(1))
            train_loss = train_acc = test_loss = top1 = top5 = None
            i += 1
            # next non-empty should be Mean training loss
            while i < len(lines) and not train_metrics_re.search(lines[i]):
                i += 1
            if i < len(lines):
                mm = train_metrics_re.search(lines[i])
                if mm:
                    train_loss = _to_float(mm.group(1))
                    train_acc = _to_float(mm.group(2))
            i += 1
            # skip Time consumption, then "Eval epoch: N"
            while i < len(lines) and "Eval epoch:" not in lines[i]:
                i += 1
            i += 1
            # Mean test loss
            while i < len(lines) and not test_loss_re.search(lines[i]):
                i += 1
            if i < len(lines):
                mt = test_loss_re.search(lines[i])
                if mt:
                    test_loss = _to_float(mt.group(1))
            i += 1
            # Top1 / Top5
            if i < len(lines):
                t1 = top1_re.search(lines[i])
                if t1:
                    top1 = _to_float(t1.group(1))
            i += 1
            if i < len(lines):
                t5 = top5_re.search(lines[i])
                if t5:
                    top5 = _to_float(t5.group(1))

            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "top1": top1,
                    "top5": top5,
                }
            )
        i += 1

    return pd.DataFrame(rows)


def get_best_from_log(log_path: Path) -> tuple:
    """Return (best_acc, best_epoch) from log.txt if present."""
    if not log_path.exists():
        return None, None
    with open(log_path, "r") as f:
        content = f.read()
    best_acc_re = re.compile(r"Best accuracy: ([\d.]+)")
    best_epoch_re = re.compile(r"Epoch number: (\d+)")
    best_acc = best_acc_re.findall(content)
    best_epoch = best_epoch_re.findall(content)
    if best_acc and best_epoch:
        return float(best_acc[-1]), int(best_epoch[-1])
    return None, None


def read_tail(log_path: Path, n_lines: int = 100) -> str:
    """Last n lines of a log file."""
    if not log_path.exists():
        return ""
    with open(log_path, "r") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:])


def is_likely_running(theme_log_path: Path, process_log_path: Path) -> bool:
    """Heuristic: training is running if process log was written recently and no 'Finished'."""
    if not process_log_path.exists():
        return False
    try:
        mtime = process_log_path.stat().st_mtime
        # consider "running" if modified in last 5 minutes
        import time
        if time.time() - mtime > 300:
            return False
    except Exception:
        return False
    with open(process_log_path, "r") as f:
        text = f.read()
    return "Finished" not in text and "Starting" in text


def load_per_class_csv(theme_dir: Path) -> Optional[pd.DataFrame]:
    """Load latest epoch*_test_each_class_acc.csv; return per-class acc and confusion (if available)."""
    pattern = str(theme_dir / "epoch*_test_each_class_acc.csv")
    files = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    if not files:
        return None
    path = Path(files[-1])
    with open(path) as f:
        first = f.readline()
    accs = [float(x) for x in first.strip().split(",")]
    return pd.DataFrame({"class_idx": range(len(accs)), "accuracy": accs})


def main():
    st.set_page_config(page_title="MMASD+ Train/Test", layout="wide")
    st.title("MMASD+ theme training & test visualization")

    work_dir = Path(
        st.sidebar.text_input("Work dir", value=str(DEFAULT_WORK_DIR))
    )
    if not work_dir.exists():
        st.error(f"Work dir not found: {work_dir}")
        st.stop()

    themes_to_show = st.sidebar.multiselect(
        "Themes",
        THEMES,
        default=THEMES,
    )
    auto_refresh = st.sidebar.checkbox("Auto-refresh (every 10s)", value=True)
    tail_lines = st.sidebar.slider("Process log tail lines", 20, 300, 80)

    # Status cards
    st.subheader("Status")
    cols = st.columns(len(themes_to_show))
    for idx, theme in enumerate(themes_to_show):
        theme_dir = work_dir / theme
        process_log = work_dir / f"{theme}.log"
        log_txt = theme_dir / "log.txt"
        df = parse_log_txt(log_txt)
        best_acc, best_epoch = get_best_from_log(log_txt)
        running = is_likely_running(log_txt, process_log)

        with cols[idx]:
            st.metric(
                theme,
                "Running" if running else ("Done" if best_acc is not None else "No data"),
                f"Epochs: {len(df)}" if not df.empty else "",
            )
            if best_acc is not None:
                st.caption(f"Best: {100*best_acc:.2f}% @ epoch {best_epoch}")

    # Plots
    st.subheader("Training & validation curves")
    all_data = []
    for theme in themes_to_show:
        log_txt = work_dir / theme / "log.txt"
        df = parse_log_txt(log_txt)
        if df.empty:
            continue
        df["theme"] = theme
        all_data.append(df)

    if not all_data:
        st.info("No log data yet. Start training and logs will appear here.")
    else:
        combined = pd.concat(all_data, ignore_index=True)

        plot_choice = st.radio(
            "Metric",
            ["Train loss", "Train acc %", "Test loss", "Top-1 %", "Top-5 %"],
            horizontal=True,
        )
        col_map = {
            "Train loss": "train_loss",
            "Train acc %": "train_acc",
            "Test loss": "test_loss",
            "Top-1 %": "top1",
            "Top-5 %": "top5",
        }
        col = col_map[plot_choice]
        if combined[col].notna().any():
            color_map = {t: THEME_COLORS[t] for t in themes_to_show if t in THEME_COLORS}
            fig = px.line(
                combined,
                x="epoch",
                y=col,
                color="theme",
                title=plot_choice,
                color_discrete_map=color_map,
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data for {plot_choice}")

        # Optional: all metrics in subplots
        with st.expander("All metrics (subplots)"):
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("Train loss", "Train acc %", "Test loss", "Top-1 %"),
                horizontal_spacing=0.08,
                vertical_spacing=0.12,
            )
            for theme in themes_to_show:
                d = combined[combined["theme"] == theme]
                if d.empty:
                    continue
                color = THEME_COLORS.get(theme, None)
                line_opt = dict(color=color) if color else {}
                fig.add_trace(
                    go.Scatter(x=d["epoch"], y=d["train_loss"], name=theme, mode="lines", line=line_opt),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(x=d["epoch"], y=d["train_acc"], name=theme, mode="lines", line=line_opt),
                    row=1,
                    col=2,
                )
                fig.add_trace(
                    go.Scatter(x=d["epoch"], y=d["test_loss"], name=theme, mode="lines", line=line_opt),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(x=d["epoch"], y=d["top1"], name=theme, mode="lines", line=line_opt),
                    row=2,
                    col=2,
                )
            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    # Per-class accuracy (latest epoch)
    st.subheader("Per-class accuracy (latest epoch)")
    theme_for_class = st.selectbox("Theme for per-class", themes_to_show)
    theme_dir = work_dir / theme_for_class
    per_class = load_per_class_csv(theme_dir)
    if per_class is not None:
        fig = px.bar(per_class, x="class_idx", y="accuracy", title=f"{theme_for_class} test accuracy per class")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No epoch*_test_each_class_acc.csv found.")

    # Process log tail
    st.subheader("Process log (tail)")
    theme_log = st.selectbox("Theme log", themes_to_show, key="tail_theme")
    process_log = work_dir / f"{theme_log}.log"
    tail = read_tail(process_log, tail_lines)
    st.text_area("Last lines", tail, height=200)

    if auto_refresh:
        import time
        st.caption("Auto-refreshing in 10s...")
        time.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()
