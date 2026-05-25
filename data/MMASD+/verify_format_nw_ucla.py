"""
Verify and compare data formats: NW-UCLA vs MMASD+ (converted).

Loads one sample from each and prints shapes/layout so we can confirm
MMASD+ conversion matches the expected NTU-style format and contrast with NW-UCLA.

Run from FreqMixFormer:
  python data/MMASD+/verify_format_nw_ucla.py
"""

import os
import os.path as osp
import json
import numpy as np

def main():
    base = osp.dirname(osp.abspath(__file__))
    proj_root = osp.abspath(osp.join(base, '../..'))  # FreqMixFormer
    os.chdir(proj_root)

    print("=" * 60)
    print("1. NW-UCLA format (data/NW-UCLA/all_sqe/*.json)")
    print("=" * 60)
    ucla_path = osp.join(proj_root, 'data/NW-UCLA/all_sqe/a01_s01_e00_v01.json')
    if not osp.isfile(ucla_path):
        print("  [SKIP] File not found:", ucla_path)
    else:
        with open(ucla_path, 'r') as f:
            ucla = json.load(f)
        skeletons = np.array(ucla['skeletons'])
        # skeletons: (T, 20, 3)
        T_ucla, V_ucla, C_ucla = skeletons.shape
        print("  JSON keys:", list(ucla.keys()))
        print("  skeletons shape: (T, V, C) = (%d, %d, %d)" % (T_ucla, V_ucla, C_ucla))
        print("  label (1-based in JSON):", ucla['label'])
        print("  Per-frame vector length (one person): %d x 3 = %d" % (V_ucla, V_ucla * C_ucla))
        print("  -> Feeder uses graph.ucla.Graph (20 nodes), M=1")

    print()
    print("=" * 60)
    print("2. MMASD+ converted npz (NTU-style)")
    print("=" * 60)
    npz_path = osp.join(base, 'MMASD+_CS.npz')
    if not osp.isfile(npz_path):
        npz_path = osp.join(proj_root, 'data/MMASD+/MMASD+_CS.npz')
    if not osp.isfile(npz_path):
        print("  [SKIP] NPZ not found. Run mmasd_to_freqmixformer_npz.py first.")
    else:
        data = np.load(npz_path)
        x_train = data['x_train']
        y_train = data['y_train']
        print("  NPZ keys:", list(data.keys()))
        print("  x_train shape: (N, T, 150) =", x_train.shape)
        print("  y_train shape: (N, num_class) =", y_train.shape)
        N, T_mmasd, dim = x_train.shape
        print("  Per-frame vector: 150 = 2 persons x 25 joints x 3")
        print("  First sample: x_train[0] shape (%d, %d)" % (T_mmasd, dim))
        # Reshape like feeder
        one = x_train[0]  # (T, 150)
        reshaped = one.reshape(T_mmasd, 2, 25, 3)
        print("  Reshape (T, 2, 25, 3):", reshaped.shape)
        print("  -> Feeder uses graph.ntu_rgb_d.Graph (25 nodes), M=2")

    print()
    print("=" * 60)
    print("3. Side-by-side summary")
    print("=" * 60)
    print("  NW-UCLA:  1 JSON per sample, (T, 20, 3), 20 joints, 1 person, 60/frame")
    print("  MMASD+:   1 npz for all,     (N,T,150),  25 joints, 2 slots, 150/frame")
    print("  Both → model input (C, T, V, M); V and M differ → different feeder/graph.")
    print("  MMASD+ correctly uses NTU pipeline (25 joints, 2 persons).")


if __name__ == '__main__':
    main()
