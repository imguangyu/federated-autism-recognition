## NVFLARE job for MMASD+ (FreqMixFormer, one-theme-per-client)

This folder contains a **minimal NVIDIA FLARE job** that runs federated training
for the FreqMixFormer model on MMASD+, with each theme (`theme1`, `theme2`, `theme3`)
as a separate client.

It is designed to work with the client code in:

- `fl_experiments/nvflare_mmasd_client.py`

### 1. Files

- `config_fed_server.json`: server-side configuration (FedAvg-like scatter-and-gather workflow).
- `config_fed_client.json`: client-side configuration (uses the Client API and launches the Python script).

You can copy this job folder, tweak hyperparameters (rounds, min clients, etc.), or
change the server aggregator to compare **different FL methods** (e.g. FedAvg vs FedProx)
while keeping the same client script.

### 2. Basic usage (simulator)

Assuming you have NVFLARE installed and `FreqMixFormer` is on your `PYTHONPATH`,
you can run the simulator by pointing it at a **job folder**:

```bash
nvflare simulator \
  -w /path/to/workspace \
  -n 3 -t 3 \
  -c theme1,theme2,theme3 \
  /absolute/path/to/FreqMixFormer/fl_experiments/nvflare_job_mmasd
```

Or use the provided runner that also auto-creates a timestamped workspace and
encodes the method into the workspace name:

```bash
bash fl_experiments/run_fl_mmasd_simulator.sh --method fedavg
bash fl_experiments/run_fl_mmasd_simulator.sh --method fedprox
```

Method-specific job folders:

- `fl_experiments/nvflare_job_mmasd_fedavg`
- `fl_experiments/nvflare_job_mmasd_fedprox`

### 3. Experiment tracking (TensorBoard)

The job streams per-client accuracy from each round to the FL server via
`SummaryWriter` and `TBAnalyticsReceiver`. Metrics are written under:

```
{workspace}/server/simulate_job/tb_events/{theme1|theme2|theme3}/
```

To view during or after a run:

```bash
# Install tensorboard if needed
pip install tensorboard

# Point to the server's tb_events folder (adjust WORKSPACE path)
tensorboard --logdir=/path/to/workspace/server/simulate_job/tb_events
```

Then open http://localhost:6006 in a browser.

