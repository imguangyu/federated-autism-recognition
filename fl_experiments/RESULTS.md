# MMASD+ / FreqMixFormer — Experiment Results

**Dataset:** MMASD+ (3 themes: theme1, theme2, theme3).  
**Model:** FreqMixFormer.  
**Settings:** 30 rounds/epochs, lr=0.1, batch_size=32, weight_decay=5e-4.

## Accuracy (test, %)

| Method | theme1 (%) | theme2 (%) | theme3 (%) | Avg (%) |
|--------|------------|------------|------------|---------|
| Local (baseline) | 99.19 | 90.67 | 97.25 | 95.70 |
| FedAvg | — | — | — | — |
| FedProx | — | — | — | — |
| FedBN | — | — | — | — |
| FedPer | — | — | — | — |

- **Local (baseline):** One model per theme, no federation (`run_local_mmasd_baseline.sh`). Best accuracy from `work_dir/baselines/local_<timestamp>/theme{1,2,3}/log.txt`.
- **FL methods:** Last-round per-client accuracy from TensorBoard in `fl_experiments/workspaces/nvflare_workspace_<method>_<timestamp>/server/simulate_job/tb_events/`.

## How to refresh this table

From the **FreqMixFormer** root directory:

```bash
pip install tensorboard   # if not already installed
python fl_experiments/collect_results.py
```

Copy the printed table into this file, or redirect:

```bash
python fl_experiments/collect_results.py >> fl_experiments/RESULTS.md
```

To view curves:

```bash
tensorboard --logdir fl_experiments/workspaces
```

## Run identifiers (for reference)

- **Local baseline:** `work_dir/baselines/local_2026-03-09_22-34-41/`
- **FL workspaces:** `fl_experiments/workspaces/nvflare_workspace_{fedavg,fedprox,fedbn,fedper}_<timestamp>/`
