# FreqMixFormer Input Format & MMASD+ Data Processing

## 1. FreqMixFormer Input Format (NTU-style)

### 1.1 File format
- **Single `.npz` file** with 4 arrays:
  - `x_train`: `(N_train, T, 150)` float32
  - `y_train`: `(N_train, num_class)` float32, **one-hot** labels
  - `x_test`: `(N_test, T, 150)` float32
  - `y_test`: `(N_test, num_class)` float32, one-hot

### 1.2 Skeleton layout in `150`
- **150** = 2 persons × 25 joints × 3 coordinates.
- Layout: `[person0_j0_x, person0_j0_y, person0_j0_z, ..., person0_j24_z, person1_j0_x, ..., person1_j24_z]`
- So each frame is **75** for one person (25×3), **150** for two persons.
- Missing person is filled with **zeros** (same as NTU).

### 1.3 Feeder reshape (NTU)
- Loaded `x` has shape `(N, T, 150)`.
- Reshape: `(N, T, 2, 25, 3)` then **transpose** to `(N, C, T, V, M)`:
  - **C=3** (coordinates x,y,z)
  - **T** (time)
  - **V=25** (joints)
  - **M=2** (persons)
- So the model expects **25 joints**, **2 persons**, **3D**.

### 1.4 Other constraints
- **Graph**: NTU uses `graph.ntu_rgb_d.Graph` (25 nodes, NTU skeleton topology).
- **num_class**: 60 for NTU60; for MMASD+ use **11** (or your number of actions).
- **window_size**: e.g. 64; sequences are cropped/resized to this length in the feeder via `valid_crop_resize`.

---

## 2. MMASD+ Data Layout

- **Root**: `data/MMASD+` → symlink to `/data/MMASD+/`
- **3D skeleton**: under `Romp_Data_All_Actions/`
  - One folder per **action** (e.g. `Arm_swing`, `Body_Swing`, `Twist_Pose`, ...).
  - Each **sample** = one subfolder (e.g. `processed_tw_41063_D1_007_y_3_1`) containing:
    - `frame_0.npz`, `frame_1.npz`, ... (one npz per frame).

### 2.1 Per-frame npz (ROMP output)
- **`joints`**: `(1, 71, 3)` float32 — **71 3D joints** (ROMP: 24 SMPL + 30 extra + 17 H36M).
- Other keys (e.g. `verts`, `global_orient`, `body_pose`, `cam_trans`) are not used for skeleton action recognition.

### 2.2 Action classes (11 total) and three themes
From the [official MMASD README](https://github.com/Li-Jicheng/MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis), actions are grouped into **3 themes**:

| Theme | Description | Actions (folder names) | Classes |
|-------|-------------|------------------------|---------|
| **theme1** | Robotic-assisted therapy | Arm_swing, Body_Swing, Chest_Expansion, Squat_Pose | 4 |
| **theme2** | Rhythm | Drumming, Marcas_Forward_Shaking, Marcas_Shaking, Sing_Clap | 4 |
| **theme3** | Yoga | Frog_Pose, Tree_Pose, Twist_Pose | 3 |

Conversion scripts support `--subset theme1`, `--subset theme2`, or `--subset theme3` to build one dataset per theme (see `data/MMASD+/README.md`).

---

## 3. How to Process MMASD+ for FreqMixFormer

### 3.1 Main steps
1. **Scan** all action folders and sample subfolders; assign **label** by action name.
2. **Load** each sample: for each frame npz read `joints` → `(1, 71, 3)`; stack along time → `(T, 71, 3)`.
3. **Map 71 → 25 joints** so the format matches NTU:
   - Option A: Use a **fixed subset** of 25 from 71 (e.g. first 25, or a predefined SMPL/H36M → NTU mapping). The script uses the first 25 joints by default; you can replace with a proper mapping.
   - Option B: Use a **custom graph and feeder** for 71 joints (and change model `num_point` and graph).
4. **Single person**: second person pad with zeros → each frame `(2, 25, 3)` → flatten to **150**.
5. **Align T**: pad or truncate to a fixed `max_num_frames` (e.g. 300 or 64) so `x` is `(N, T, 150)`.
6. **Train/test split**: e.g. 80/20 random (or by subject ID if you parse it from the path).
7. **Labels**: class index 0..10 (or 0..num_class-1), then **one-hot** to `(N, num_class)`.
8. **Save** `npz`: `x_train`, `y_train`, `x_test`, `y_test`.

### 3.2 Config / feeder
- **Feeder**: Use `feeders.feeder_ntu.Feeder` (expects `(N, T, 150)` and reshape to `(N, 3, T, 25, 2)`).
- **Model**: `num_class=11`, `num_point=25`, `num_person=2`, `graph=graph.ntu_rgb_d.Graph`.
- **Data path**: point `data_path` to the generated npz (e.g. `data/MMASD+/MMASD+_CS.npz` or similar).

### 3.3 Joint mapping 71 → 25
- **`--joint-map smpl24` (default)**: Use ROMP’s first **24** joints as SMPL 24, then duplicate pelvis (index 0) as the 25th so the output matches NTU’s 25-joint format. This keeps standard SMPL body joints.
- **`--joint-map first25`**: Use ROMP joints 0..24 (first 25 of 71).
- For a custom NTU↔SMPL mapping you can edit `SMPL24_TO_NTU25_ROMP_INDICES` in the script.

### 3.4 Run conversion
From `FreqMixFormer` directory:
```bash
# All 11 classes
python data/MMASD+/mmasd_to_freqmixformer_npz.py --root /data/MMASD+/Romp_Data_All_Actions --out data/MMASD+/MMASD+_CS.npz

# Per-theme subsets (4, 4, 3 classes)
python data/MMASD+/mmasd_to_freqmixformer_npz.py --subset theme1 --out data/MMASD+/MMASD+_theme1.npz
python data/MMASD+/mmasd_to_freqmixformer_npz.py --subset theme2 --out data/MMASD+/MMASD+_theme2.npz
python data/MMASD+/mmasd_to_freqmixformer_npz.py --subset theme3 --out data/MMASD+/MMASD+_theme3.npz

# Or use first 25 of 71: --joint-map first25
python data/MMASD+/mmasd_to_freqmixformer_npz.py --joint-map first25 --out data/MMASD+/MMASD+_CS.npz
```
Then train with config `config/mmasd/joint.yaml` (e.g. `python main.py --config config/mmasd/joint.yaml`).

---

## 5. Visualization (verify correctness)

Use `data/MMASD+/visualize_skeleton.py` to check skeletons from the converted npz or from raw folders.

**From converted npz** (3 frames: 0, 50, 100):
```bash
cd FreqMixFormer
python data/MMASD+/visualize_skeleton.py --npz data/MMASD+/MMASD+_CS.npz --sample 0 --frames 0,50,100 --out data/MMASD+/vis_npz.png
```

**From raw MMASD+ folder** (SMPL24 mapping):
```bash
python data/MMASD+/visualize_skeleton.py --raw /data/MMASD+/Romp_Data_All_Actions/Twist_Pose/Twist_pose/processed_tw_41063_D1_007_y_3_1 --joint-map smpl24 --out data/MMASD+/vis_raw.png
```

**Animate one sample** (requires `pip install imageio`):
```bash
python data/MMASD+/visualize_skeleton.py --npz data/MMASD+/MMASD+_CS.npz --sample 0 --animate --out data/MMASD+/vis_anim.gif
```

---

## 4. Summary Table

| Item        | NTU (FreqMixFormer) | MMASD+ (raw) | After processing for FreqMixFormer   |
|------------|----------------------|--------------|---------------------------------------|
| Storage    | One .npz             | Per-frame .npz in folders | One .npz (x_train, y_train, x_test, y_test) |
| Joints     | 25                   | 71           | 25 (subset/mapping from 71)           |
| Persons    | 2                    | 1            | 2 (second person zeros)               |
| Frame dim  | 150                  | 71×3 per frame | 150                                 |
| Labels     | One-hot (e.g. 60)    | Folder name  | One-hot (11)                          |

---

## 6. Comparison with NW-UCLA (verification)

NW-UCLA is another skeleton dataset already supported by FreqMixFormer. Comparing formats helps verify MMASD+ conversion.

### 6.1 NW-UCLA format

| Aspect | NW-UCLA | MMASD+ (converted) |
|--------|---------|--------------------|
| **Storage** | One JSON per sample: `data/NW-UCLA/all_sqe/<file_name>.json` | One npz: `x_train`, `y_train`, `x_test`, `y_test` |
| **JSON keys** | `file_name`, `skeletons`, `label` | — |
| **Skeletons** | List of frames; each frame = **20 joints** × [x,y,z] → shape (T, 20, 3) | (N, T, 150) → (T, 2, 25, 3) per sample |
| **Joints (V)** | **20** | **25** |
| **Persons (M)** | 1 (single person) | 2 (second zero-padded) |
| **Time (T)** | Variable per sample | Fixed max_frames (300), then feeder crops to window_size |
| **Label** | In JSON, 1-based (1–10); feeder uses 0-based | One-hot in npz, 11 classes |
| **Feeder** | `feeders.feeder_ucla.Feeder` (reads JSON, outputs C,T,V,1) | `feeders.feeder_ntu.Feeder` (reads npz, reshape to N,C,T,V,M) |
| **Graph** | `graph.ucla.Graph` (20 nodes) | `graph.ntu_rgb_d.Graph` (25 nodes) |
| **Per-frame vector** | 20×3 = **60** (one person) | 2×25×3 = **150** (two person slots) |

### 6.2 Takeaways

- **NW-UCLA**: 20 joints, 1 person, one JSON per sequence, variable T; different feeder and graph.
- **MMASD+ (converted)**: 25 joints, 2 person slots (second zeros), single npz, fixed T then windowed; uses NTU feeder and graph.
- Both feed into the same model layout **(C, T, V, M)**; V and M differ (20 vs 25, 1 vs 2), so **MMASD+ uses the NTU pipeline**, not the UCLA one.

### 6.3 Run format verification script

From `FreqMixFormer`:
```bash
python data/MMASD+/verify_format_nw_ucla.py
```
This loads one NW-UCLA JSON and one MMASD+ npz sample and prints shapes and layout for comparison.

---

## 7. MMASD+ in NW-UCLA style (20 joints, 1 person)

You can also store MMASD+ in the same layout as NW-UCLA and use the UCLA feeder/graph (20 joints, 1 person).

### 7.1 Conversion to UCLA format

Run from `FreqMixFormer`:
```bash
python data/MMASD+/mmasd_to_ucla_format.py --root /data/MMASD+/Romp_Data_All_Actions --out-dir data/MMASD+
```
Options: `--joint-map smpl24` (default) or `first25`, `--train-ratio 0.8`, `--seed 42`.

This writes:
- `data/MMASD+/all_sqe/<file_name>.json` — one JSON per sample with `skeletons` (T, 20, 3) and `label` (1-based 1..11).
- `data/MMASD+/train_data_dict.json` and `data/MMASD+/val_data_dict.json` — lists of `{ file_name, length, label }`.

Joints: 25 (SMPL24 or first25) from raw → first 20 kept to match UCLA.

### 7.2 Config and training

Use config **`config/mmasd/joint_ucla.yaml`**:
- Feeder: `feeders.feeder_mmasd.Feeder` (loads data_dict from JSON and reads each sample from `all_sqe/`).
- Model: `num_point=20`, `num_person=1`, `graph=graph.ucla.Graph`.
- Set `data_path` to the directory that contains `all_sqe/`, `train_data_dict.json`, and `val_data_dict.json` (e.g. `data/MMASD+`).

Example:
```bash
python main.py --config config/mmasd/joint_ucla.yaml
```
