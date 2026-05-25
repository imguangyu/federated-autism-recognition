#!/usr/bin/env python3
"""
Extract APFL alpha (per round, per theme) from TensorBoard events to a CSV file.

Usage (from FreqMixFormer root):

  # Use latest APFL workspace under fl_experiments/workspaces, write to fl_experiments/logs/apfl_alpha.csv
  python fl_experiments/extract_apfl_alpha_csv.py

  # Specify workspace and output
  python fl_experiments/extract_apfl_alpha_csv.py --workspace fl_experiments/workspaces/nvflare_workspace_apfl_2026-03-14_20-50-08 --output apfl_alpha.csv

  # Multiple workspaces (e.g. multiple runs): pass multiple --workspace; run name is taken from dir name
  python fl_experiments/extract_apfl_alpha_csv.py --workspace ws1 --workspace ws2 --output alphas.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACES_ROOT = SCRIPT_DIR / "workspaces"
THEMES = ("theme1", "theme2", "theme3")


def load_alpha_rounds(run_name: str, ws_dir: Path) -> List[Dict]:
    """Load per-round alpha from TensorBoard events (APFL). Returns list of dicts."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise SystemExit("Need tensorboard: pip install tensorboard")

    tb_root = ws_dir / "server" / "simulate_job" / "tb_events"
    if not tb_root.exists():
        return []

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
        for s in ea.Scalars(tag):
            rows.append({"run": run_name, "theme": theme, "round": s.step, "alpha": s.value})
    return rows


def find_latest_apfl_workspace() -> Optional[Path]:
    """Return path to the latest nvflare_workspace_apfl_* directory under WORKSPACES_ROOT."""
    if not WORKSPACES_ROOT.exists():
        return None
    candidates = sorted(WORKSPACES_ROOT.glob("nvflare_workspace_apfl_*"), reverse=True)
    for p in candidates:
        if p.is_dir() and (p / "server" / "simulate_job" / "tb_events").exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract APFL alpha over rounds to CSV")
    parser.add_argument(
        "--workspace",
        type=Path,
        action="append",
        default=[],
        help="APFL workspace path(s). If not set, use latest under fl_experiments/workspaces",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=SCRIPT_DIR / "logs" / "apfl_alpha.csv",
        help="Output CSV path (default: fl_experiments/logs/apfl_alpha.csv)",
    )
    args = parser.parse_args()

    workspaces: List[tuple[str, Path]] = []  # (run_name, path)
    if args.workspace:
        for wp in args.workspace:
            wp = wp.resolve()
            if not wp.is_dir():
                raise SystemExit(f"Workspace not a directory: {wp}")
            run_name = wp.name  # e.g. nvflare_workspace_apfl_2026-03-14_20-50-08
            workspaces.append((run_name, wp))
    else:
        latest = find_latest_apfl_workspace()
        if latest is None:
            raise SystemExit(
                "No APFL workspace found. Run APFL first or pass --workspace /path/to/nvflare_workspace_apfl_*"
            )
        workspaces.append((latest.name, latest.resolve()))

    all_rows: List[Dict] = []
    for run_name, ws_dir in workspaces:
        rows = load_alpha_rounds(run_name, ws_dir)
        if not rows:
            print(f"Warning: no alpha data in {ws_dir}")
            continue
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No alpha data found in any workspace.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "theme", "round", "alpha"])
        w.writeheader()
        for r in sorted(all_rows, key=lambda x: (x["run"], x["theme"], x["round"])):
            w.writerow(r)

    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
