# MMASD+ Skeleton Data

MMASD (Multimodal Dataset for Autism Intervention Analysis) and its extension MMASD+ provide skeleton sequences from intervention sessions. The dataset organizes **11 actions** into **3 themes** (from the [official MMASD README](https://github.com/Li-Jicheng/MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis)).

## Themes and actions (folder names)

| Theme | Description | Action folders (in `Romp_Data_All_Actions/`) |
|-------|-------------|---------------------------------------------|
| **theme1** | Robotic-assisted therapy | `Arm_swing`, `Body_Swing`, `Chest_Expansion`, `Squat_Pose` |
| **theme2** | Rhythm | `Drumming`, `Marcas_Forward_Shaking`, `Marcas_Shaking`, `Sing_Clap` |
| **theme3** | Yoga | `Frog_Pose`, `Tree_Pose`, `Twist_Pose` |

- **theme1**: 4 actions | **theme2**: 4 actions | **theme3**: 3 actions  

Conversion scripts support `--subset theme1`, `--subset theme2`, or `--subset theme3` to build one npz (or UCLA-format output) per theme. **Labels are global 0..10** (same order as full 11 actions from `get_action_folders`), so **num_class is always 11** for every subset; the model output dimension stays 11.

## Data layout

- Raw ROMP output: `Romp_Data_All_Actions/<ActionFolder>/<sample_dir>/frame_*.npz`  
  (e.g. `Arm_swing/processed_Arm_swingas_20583_D1_000_y_6_1/frame_0.npz`)  
- Each `frame_*.npz` contains `joints` (1, 71, 3) and `pj2d_org` (1, 71, 2).

See `INPUT_FORMAT_AND_MMASD.md` for FreqMixFormer format and conversion commands.
