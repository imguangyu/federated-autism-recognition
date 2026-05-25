#!/usr/bin/env python3
"""
Collect FL and local baseline results into a single table (printed as Markdown).
- Central local baseline (standalone): parsed from work_dir/baselines/*/theme*/log.txt (Best accuracy).
- FL methods (including "local" under the FL framework): read from TensorBoard events in
  fl_experiments/workspaces (last-round per-client accuracy).

Requires: pip install tensorboard
"""
import os
import re
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_ROOT = os.path.join(PROJ_ROOT, "work_dir", "baselines")
WORKSPACES_ROOT = os.path.join(PROJ_ROOT, "fl_experiments", "workspaces")
THEMES = ("theme1", "theme2", "theme3")


def collect_local_baseline():
    out = {}
    if not os.path.isdir(BASELINES_ROOT):
        return out
    for run_dir in sorted(os.listdir(BASELINES_ROOT)):
        if not run_dir.startswith("local_"):
            continue
        path = os.path.join(BASELINES_ROOT, run_dir)
        if not os.path.isdir(path):
            continue
        out["run"] = run_dir
        for theme in THEMES:
            log = os.path.join(path, theme, "log.txt")
            if not os.path.isfile(log):
                continue
            with open(log) as f:
                content = f.read()
            m = re.search(r"Best accuracy:\s*([\d.]+)", content)
            if m:
                out[theme] = round(float(m.group(1)) * 100, 2)
        break  # use latest run only
    return out


def collect_fl_from_tb():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return {}
    # FL methods, including "local" which is local-only training under the FL framework
    methods = ["fedavg", "fedprox", "fedbn", "fedper", "local"]
    out = {}
    for m in methods:
        dirs = [d for d in os.listdir(WORKSPACES_ROOT) if d.startswith("nvflare_workspace_" + m + "_")]
        if not dirs:
            continue
        dirs.sort(reverse=True)
        wp = os.path.join(WORKSPACES_ROOT, dirs[0], "server", "simulate_job", "tb_events")
        if not os.path.isdir(wp):
            continue
        out[m] = {}
        for theme in THEMES:
            tp = os.path.join(wp, theme)
            if not os.path.isdir(tp):
                continue
            evs = [os.path.join(tp, f) for f in os.listdir(tp) if f.startswith("events.")]
            if not evs:
                continue
            ea = event_accumulator.EventAccumulator(evs[0])
            ea.Reload()
            if "scalars" not in ea.Tags() or not ea.Tags()["scalars"]:
                continue
            tags = [t for t in ea.Tags()["scalars"] if "accuracy" in t]
            if not tags:
                continue
            vals = ea.Scalars(tags[0])
            if vals:
                out[m][theme] = round(vals[-1].value * 100, 2)
    return out


def main():
    os.chdir(PROJ_ROOT)
    local = collect_local_baseline()
    fl = collect_fl_from_tb()

    # Build table
    rows = []
    # Header
    cols = ["Method", "theme1 (%)", "theme2 (%)", "theme3 (%)", "Avg (%)"]
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("|" + "|".join(["---"] * len(cols)) + "|")

    # Central local baseline (standalone)
    if local:
        t1 = local.get("theme1", "—")
        t2 = local.get("theme2", "—")
        t3 = local.get("theme3", "—")
        if isinstance(t1, (int, float)) and isinstance(t2, (int, float)) and isinstance(t3, (int, float)):
            avg = round((t1 + t2 + t3) / 3, 2)
        else:
            avg = "—"
        rows.append("| Local (standalone) | {} | {} | {} | {} |".format(t1, t2, t3, avg))
    else:
        rows.append("| Local (standalone) | — | — | — | — |")

    for method in ["fedavg", "fedprox", "fedbn", "fedper", "local"]:
        if method not in fl or not fl[method]:
            rows.append("| {} | — | — | — | — |".format(method))
            continue
        d = fl[method]
        t1 = d.get("theme1", "—")
        t2 = d.get("theme2", "—")
        t3 = d.get("theme3", "—")
        if isinstance(t1, (int, float)) and isinstance(t2, (int, float)) and isinstance(t3, (int, float)):
            avg = round((t1 + t2 + t3) / 3, 2)
        else:
            avg = "—"
        rows.append("| {} | {} | {} | {} | {} |".format(method, t1, t2, t3, avg))

    table = "\n".join(rows)
    print(table)
    if not fl and "tensorboard" not in sys.modules:
        print("\n# FL numbers require: pip install tensorboard", file=sys.stderr)


if __name__ == "__main__":
    main()
