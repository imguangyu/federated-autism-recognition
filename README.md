# Federated Autism Recognition

Skeleton-based action recognition on **MMASD+** using [FreqMixFormer](https://github.com/wenhanwu95/FreqMixFormer), with **local** and **federated** training across theme-based clients.

## Setup

```bash
conda env create -f environment.yml   # or: bash setup_env.sh
conda activate freqmixformer
pip install -r requirements.txt
```

For federated experiments, also install NVFLARE and TensorBoard:

```bash
pip install nvflare tensorboard
```

## MMASD+ data

Large artifacts (`.npz`, converted JSON trees) are not in the repo. See `data/MMASD+/README.md` for themes and conversion.

From the project root, build theme subsets (example):

```bash
python data/MMASD+/mmasd_to_freqmixformer_npz.py --subset theme1 --out data/MMASD+/MMASD+_theme1.npz
python data/MMASD+/mmasd_to_freqmixformer_npz.py --subset theme2 --out data/MMASD+/MMASD+_theme2.npz
python data/MMASD+/mmasd_to_freqmixformer_npz.py --subset theme3 --out data/MMASD+/MMASD+_theme3.npz
```

## Local training (per theme)

```bash
# Parallel on GPUs 0,1,2 (uses CUDA_VISIBLE_DEVICES)
bash run_mmasd_themes.sh

# Or timestamped baseline (recommended)
bash fl_experiments/run_local_mmasd_baseline.sh
```

Configs: `config/mmasd/joint_theme{1,2,3}.yaml` (11 global action classes).

Monitor:

```bash
streamlit run streamlit_mmasd_vis.py
```

## Federated learning

See `fl_experiments/README.md` for NVFLARE jobs (FedAvg, FedProx, FedBN, FedPer, APFL, local baseline, 5×6 splits).

```bash
bash fl_experiments/run_fl_mmasd_simulator.sh --method fedavg
streamlit run fl_experiments/streamlit_fl_methods_vis.py
```

## Repository layout

| Path | Description |
|------|-------------|
| `config/mmasd/` | MMASD+ / theme training configs |
| `data/MMASD+/` | Conversion scripts, docs, split JSON |
| `fl_experiments/` | NVFLARE jobs, run scripts, FL Streamlit |
| `feeders/feeder_mmasd.py` | MMASD+ data loader |
| `run_mmasd_themes.sh` | Local multi-theme training launcher |

## Citation

If you use the underlying FreqMixFormer model, cite:

```bibtex
@inproceedings{wu2024frequencyguidancemattersskeletal,
  author = {Wenhan Wu and Ce Zheng and Zihao Yang and Chen Chen and Srijan Das and Aidong Lu},
  title = {Frequency Guidance Matters: Skeletal Action Recognition by Frequency-Aware Mixed Transformer},
  booktitle = {ACM Multimedia 2024},
  year = {2024}
}
```

## Acknowledgements

Built on [FreqMixFormer](https://github.com/wenhanwu95/FreqMixFormer), [Skeleton-MixFormer](https://github.com/ElricXin/Skeleton-MixFormer), and [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) data tooling.
