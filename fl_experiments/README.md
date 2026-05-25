## FL experiments for MMASD+ (FreqMixFormer)

This folder contains code and notes for **federated learning (FL) experiments** using the FreqMixFormer model on the **MMASD+** dataset, with each *theme* treated as a separate client.

- **Local training baseline**: train one model per theme (theme1–3), no aggregation.
- **FL methods**: run **NVIDIA FLARE** with one theme per client and compare methods (currently: **FedAvg**, **FedProx**).

### 1. Local baseline (centralized per-theme training)

From the `FreqMixFormer` project root:

```bash
# Recommended: timestamped outputs (won't overwrite prior runs)
bash fl_experiments/run_local_mmasd_baseline.sh

# Legacy: overwrites work_dir/mmasd/theme{1,2,3}
bash run_mmasd_themes.sh
```

You can then evaluate specific checkpoints using `evaluate.sh` (adapt paths and weights as needed).

### 2. Federated learning (NVFLARE simulator)

Install NVFLARE and TensorBoard for experiment tracking:

```bash
pip install nvflare tensorboard
```

Run FL with a method-specific, timestamped workspace (workspace name reflects the method):

```bash
# FedAvg
bash fl_experiments/run_fl_mmasd_simulator.sh --method fedavg

# FedProx (default mu is set in the job config)
bash fl_experiments/run_fl_mmasd_simulator.sh --method fedprox
```

By default, the simulator creates workspaces under:

```
fl_experiments/workspaces/nvflare_workspace_{fedavg|fedprox}_YYYY-MM-DD_HH-MM-SS/
```

### 3. Results / tracking

- **Local baseline artifacts**: `work_dir/baselines/local_YYYY-MM-DD_HH-MM-SS/theme{1,2,3}/`
- **FL TensorBoard logs**:

```
fl_experiments/workspaces/nvflare_workspace_{method}_*/server/simulate_job/tb_events/{theme1|theme2|theme3}/
```

To view:

```bash
tensorboard --logdir fl_experiments/workspaces/nvflare_workspace_fedavg_*/server/simulate_job/tb_events
```

