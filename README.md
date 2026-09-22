# SPARC — 3D CT from a handful of X-ray projections

SPARC is a foundation model that reconstructs a three-dimensional CT volume from four or eight X-ray projections. It is pretrained on 47,149 chest CT volumes with randomised acquisition geometry, so its encoder knows where each projection was taken. This repository provides the code, the pretrained weights, the sixteen per-dataset reconstruction models behind the paper's results and the data splits.

[Pretrained weights](https://huggingface.co/lyqun/SPARC) · [Reconstruction models](https://huggingface.co/lyqun/SPARC-reconstruction) · [Data splits](splits/)

**A foundation model recovers three-dimensional anatomy and clinical findings from sparse X-ray projections**
Yiqun Lin, Jiayang Xu, Lie Ju and colleagues · Manuscript (2026)

## Highlights

- One pretrained backbone, adapted to eight CT datasets and three anatomies it never saw in pretraining (head, dental, spine).
- With four or eight projections, SPARC gave the highest PSNR and SSIM in all sixteen dataset and view-count settings against five competing methods retrained under the same protocol.
- Each projection is tagged with the geometry of its rays (Plücker coordinates), so the same model handles different view counts, angles and source-detector distances.
- The paper's reconstruction results on all eight datasets can be reproduced from the released weights by inference alone.

![SPARC: projections are encoded with their ray geometry, lifted into a shared 3D feature volume and decoded at any resolution.](images/method-overview.png)

*Figure 1. The SPARC model. Each projection is encoded together with the geometry of its rays, sampled into a 16³ feature volume that a 3D transformer refines, and decoded into Hounsfield units at any query point. During pretraining a second head predicts features of a frozen CT encoder.* [View full-size image](images/method-overview.png)

## Getting started

### 1. Install

```bash
git clone https://github.com/HORIZONHealthcare/SPARC.git
cd SPARC
conda create -n sparc python=3.11 -y
conda activate sparc
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Get model weights and data

| Resource | Link | Use |
|---|---|---|
| Pretrained backbone | [Model card](https://huggingface.co/lyqun/SPARC) · [Checkpoint file](https://huggingface.co/lyqun/SPARC/blob/main/sparc_stage2_backbone.pth) | Starting point for reconstruction on a new dataset |
| Stage-1 CT encoder | [Model card](https://huggingface.co/lyqun/SPARC) · [Checkpoint file](https://huggingface.co/lyqun/SPARC/blob/main/sparc_stage1_ctmae.pth) | Only needed to rerun Stage-2 pretraining |
| Reconstruction models | [Model card](https://huggingface.co/lyqun/SPARC-reconstruction) | Sixteen checkpoints, one per dataset and view count (4 or 8 projections) |
| Data splits | [`splits/`](splits/) | The reconstruction train, validation and test lists used in the paper, after removing test volumes that repeat a training scan |

The weights are released under CC BY-NC-SA 4.0. Accept the terms on the model page, log in with `hf auth login`, then download into `weights/`:

```bash
hf download lyqun/SPARC sparc_stage2_backbone.pth --local-dir weights
hf download lyqun/SPARC-reconstruction cq500_v8.pth --local-dir weights
```

The reconstruction models are named `<dataset>_v<views>.pth`, with `<dataset>` one of `ctrate`, `totalsegmentator`, `msd`, `abdomenct1k`, `amos`, `cq500`, `toothfairy3`, `verse`.

The datasets are not redistributed. Download them from their sources:

| Dataset | Source | Licence |
|---|---|---|
| CT-RATE (chest; pretraining, chest reconstruction) | [link](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | CC BY-NC-SA 4.0 |
| TotalSegmentator | [link](https://doi.org/10.5281/zenodo.6802613) | CC BY 4.0 |
| Medical Segmentation Decathlon (six CT tasks) | [link](http://medicaldecathlon.com) | CC BY-SA 4.0 |
| AbdomenCT-1K | [link](https://github.com/JunMa11/AbdomenCT-1K) | see source |
| AMOS (CT) | [link](https://doi.org/10.5281/zenodo.7262581) | CC BY 4.0 |
| CQ500 (head) | [link](http://headctstudy.qure.ai/dataset) | CC BY-NC-SA 4.0 |
| ToothFairy3 (dental CBCT) | [link](https://toothfairy3.grand-challenge.org/dataset/) | CC BY-NC-SA 4.0, registration |
| VerSe (spine) | [link](https://osf.io/nqjyw/), [link](https://osf.io/t98fz/) | CC BY-SA 4.0 |

#### Data preparation

Every loader reads per-volume zarr arrays - Hounsfield units clipped to
`[-1024, 1024]`, with voxel spacing and a body bounding box stored as
attributes - listed in single-column CSV manifests whose column is `ct_path`.
The converters in `preprocessing/` write that schema:

| Script | Input |
| --- | --- |
| `preprocess_ctrate_zarr.py` | CT-RATE NIfTI, whose files hold raw stored values and an identity affine: real HU and spacing are recovered from the metadata CSV |
| `preprocess_ts_zarr.py` | TotalSegmentator NIfTI (already real HU and spacing) |
| `preprocess_nii_zarr.py` | any other NIfTI dataset, driven by a manifest of source paths (MSD, AMOS, AbdomenCT-1K, VerSe, ToothFairy3) |
| `preprocess_dicom_zarr.py` | DICOM studies, one study per volume (CQ500) |
| `build_totalseg_manifest.py` | builds a TotalSegmentator source manifest |

The volumes themselves are not redistributed: each dataset has its own licence and access procedure
(links in the table above). The split files in [`splits/`](splits/) list which volumes the paper used.

Copy the split files into place with `cp -r splits "$DATA_ROOT/"`. Each split lists volumes as `processed/<dataset>_zarr/<case>.zarr`, relative to `DATA_ROOT`, so write each dataset's converted volumes to that folder (for example `$DATA_ROOT/processed/cq500_zarr/`). The folder names are listed in [`splits/README.md`](splits/README.md).

### 3. Reconstruct and evaluate with the released models

```bash
export DATA_ROOT=/path/to/data        # holds splits/ and processed/
export OUTPUT_ROOT=/path/to/outputs
export PYTHONPATH=.
python finetune_foundation_recon.py --config configs/ours_head_converged_v8.yaml \
    --init_ckpt weights/sparc_stage2_backbone.pth --eval_ckpt weights/cq500_v8.pth \
    --metrics_manifest "$DATA_ROOT/splits/cq500_test_zarr.csv" \
    --metrics_csv cq500_V8_ours.csv
```

This writes the PSNR and SSIM of every test volume; over the 71 CQ500 test volumes they average 33.06 dB and 0.9689, the values reported in the paper. Replace `--metrics_csv` with `--export_dir <folder>` to save the reconstructed volumes (and their ground truth) as NumPy arrays in Hounsfield units. The config for each released model:

| Model | Config | Test split |
|---|---|---|
| `ctrate_v{4,8}` | `recon_ours_clssplit_converged_v{4,8}.yaml` | `cls3039_test_zarr.csv` |
| `totalsegmentator_v{4,8}` | `recon_ours_ts_converged_v{4,8}.yaml` | `ts_seg_test.csv` |
| `msd_v{4,8}` | `ours_msd_full_converged_v{4,8}.yaml` | `msd_full_test_zarr.csv` |
| `abdomenct1k_v{4,8}` | `ours_abdomenct1k_converged_v{4,8}.yaml` | `abdomenct1k_test_zarr.csv` |
| `amos_v{4,8}` | `ours_amos_ct_converged_v{4,8}.yaml` | `amos_ct_test_zarr.csv` |
| `cq500_v{4,8}` | `ours_head_converged_v{4,8}.yaml` | `cq500_test_zarr.csv` |
| `toothfairy3_v{4,8}` | `ours_tf3_converged_v{4,8}.yaml` | `toothfairy3_test_zarr.csv` |
| `verse_v{4,8}` | `ours_verse_converged_v{4,8}.yaml` | `verse_test_zarr.csv` |

Each view count is a separate model: evaluate a `v8` model with eight projections only.

To adapt SPARC to your own dataset, copy one of these configs, point its manifests at your splits, and train from the pretrained backbone with `--init_ckpt weights/sparc_stage2_backbone.pth` (see below).

### 4. Train

```bash
export DATA_ROOT=/path/to/data        # holds splits/*.csv and the volumes they list
export OUTPUT_ROOT=/path/to/outputs   # checkpoints, logs, metric CSVs
export PYTHONPATH=.
```

Configs address data and runs through `${DATA_ROOT}` and `${OUTPUT_ROOT}`, which
are expanded when a config is loaded. Paths inside a manifest may be absolute or
relative to `DATA_ROOT`.

Pretraining (single node shown; the paper's Stage 2 ran on 16 GPUs, one sample
per GPU, and the learning rate is deliberately **not** scaled with world size):

```bash
torchrun --standalone --nproc_per_node=4 train_ctmae.py      --config configs/ctmae_pretrain_crop.yaml
torchrun --standalone --nproc_per_node=4 train_foundation.py --config configs/stage2_foundation.yaml
```

Stage 2 reads the frozen Stage-1 encoder from `target.ctmae_init_path` in its config. To skip
Stage 1, download `sparc_stage1_ctmae.pth` and set that entry to its path.

Reconstruction, one run per dataset and view count, starting from the pretrained backbone
(`--init_ckpt` overrides the `init_ckpt` entry of the config, which points at your own Stage-2 run):

```bash
python finetune_foundation_recon.py --config configs/ours_head_converged_v8.yaml \
    --init_ckpt weights/sparc_stage2_backbone.pth
```

Configs are named `ours_<dataset>_converged_v{4,8}.yaml`, plus
`recon_ours_{clssplit,ts}_converged_v{4,8}.yaml` for the CT-RATE and
TotalSegmentator splits that the downstream tasks reuse. Each view count is a
separate model: a `v8` checkpoint is never evaluated at `V=4`.

Every reconstruction config runs a fixed protocol: at most 400 epochs, early
stopping on held-out validation PSNR (minimum 120 epochs, then five consecutive
validations without a 0.02 dB improvement), validating every 20 epochs, and the
best-validation checkpoint is the one evaluated on test.

To re-evaluate a finished run and write per-volume PSNR and SSIM:

```bash
python finetune_foundation_recon.py --config configs/ours_head_converged_v8.yaml \
    --eval_ckpt best.pt \
    --metrics_manifest $DATA_ROOT/splits/cq500_test_zarr.csv \
    --metrics_csv $OUTPUT_ROOT/metrics/cq500_V8_ours.csv
```

Downstream probes attach a head to the pretrained encoder:

```bash
python train_repr_cls.py        --config configs/reprB_cls_pre_lp_v8.yaml   # linear probe
python train_repr_downstream.py --config configs/reprB_seg_pre_ft_v8.yaml   # finetune
```

## Environment and hardware

| Component | Version or requirement |
|---|---|
| Python | 3.11 |
| Framework | PyTorch 2.5.1; nanodrr 0.1.4; see `requirements.txt` |
| CUDA / CPU | CUDA 12.4 |
| Hardware and GPU memory | Inference: one GPU with at least 8 GB (peak 5.7 GiB at a 256³ output grid, about 4 s per volume on an NVIDIA GH200). Training: the paper's models were trained on NVIDIA GH200 GPUs (96 GB); Stage 2 used 16 GPUs with one volume each, and per-dataset reconstruction used one GPU with three volumes per step. |

## Repository layout

```
src/models/foundation.py         SparseViewFoundation: one backbone, two heads
src/models/xray_encoder_2d.py    per-view ConvNeXt pyramid + Plucker injection
src/models/feature_volume_3d.py  projection, max-fuse, 3D transformer -> F
src/models/point_decoder.py      implicit point decoder (resolution-free)
src/models/ct_mae.py             Stage-1 3D masked autoencoder
src/datasets/drr_dataset.py      CT loading and acquisition geometry
src/utils/nanodrr_helpers.py     differentiable DRR rendering and ray geometry
src/utils/plucker.py             Plucker ray maths
src/utils/project_paths.py       ${DATA_ROOT} / ${OUTPUT_ROOT} resolution
src/utils/train_common.py        epoch budget, early stopping, exact resume

preprocessing/                   CT volumes -> per-volume zarr + CSV manifests

train_ctmae.py                   Stage 1
train_foundation.py              Stage 2
finetune_foundation_recon.py     Stage 3, reconstruction
train_repr_cls.py                Stage 3, classification head
train_repr_downstream.py         Stage 3, segmentation head
```

## Notes on scope

This repository contains SPARC only. The baseline reconstructors we compare
against (DIF-Net, C2RV, DIF-Gaussian, SVCT, DeepSparse) are third-party code and
are not redistributed here; please obtain them from their original authors. Where
our modules build on ideas from those works - notably the projection-and-sample
lifting and the point decoder, which follow DeepSparse - the source is noted in
the file headers.

## Citation

If you use this code, the weights or the splits, please cite:

```bibtex
@misc{lin2026sparc,
  title  = {A foundation model recovers three-dimensional anatomy and clinical findings from sparse X-ray projections},
  author = {Lin, Yiqun and Xu, Jiayang and Ju, Lie and Wang, Hualiang and Guo, Jiarong and Yao, Huifeng and Sun, Haoran and Zhou, Yukun},
  year   = {2026},
  note   = {Manuscript}
}
```

Please also cite the datasets you use (see the table above).

## Contact

For research enquiries, contact [Yukun Zhou](mailto:yukun.zhou.19@ucl.ac.uk) or [Yiqun Lin](mailto:yiqun.lin@ucl.ac.uk).

## License

The code is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0); see [LICENSE](LICENSE). The model weights on Hugging Face are released under CC BY-NC-SA 4.0, because the pretraining data (CT-RATE) and two of the evaluation datasets (CQ500, ToothFairy3) carry that licence. Commercial use is not permitted. The datasets keep their own licences, which these do not override.
