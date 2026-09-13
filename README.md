# SPARC

Reference implementation of **SPARC**, a geometry-aware foundation model for
sparse-view CT reconstruction: a handful of X-ray projections in, a full 3D
volume out, with an encoder whose features also transfer to downstream
recognition without reconstructing anything at inference.

> Y. Lin, J. Xu, L. Ju, H. Wang, J. Guo, H. Yao, H. Sun, Y. Zhou.
> *A foundation model recovers three-dimensional anatomy and clinical findings
> from sparse X-ray projections.*

## Method

Three stages:

**Stage 1 - CT semantic target.** A 3D masked autoencoder (ViT-L/16, 0.75 mask
ratio) is trained on CT volumes alone. Its decoder is discarded and its encoder
frozen; it maps a CT to `16^3 = 4096` anatomy-aware tokens that serve as the
prediction target in Stage 2.

**Stage 2 - sparse-view backbone.** Projections are rendered online with
randomised acquisition geometry (view count `V` in `[2,16]`; uniform-circular,
random-circular, limited-angle, bi-planar and clustered patterns; randomised
source/detector distances). Each view passes through a shared ConvNeXt pyramid
with a transformer at the coarsest scale, into which the 6-D Plucker coordinate
of every ray is injected, so the encoder reasons about *where* a projection was
taken rather than treating views as an unordered set. A `16^3` grid of points is
projected into all views, bilinearly sampled, max-fused across views and refined
by a 3D transformer into the feature volume `F` of shape `16^3 x 768`. Two heads
consume `F`: an implicit point decoder supervised by voxel reconstruction
(primary), and a light head that predicts the frozen CT-MAE tokens under a
smooth-L1 cross-modal joint-embedding loss (auxiliary, weight `lambda = 1`). The
semantic head is discarded afterwards.

**Stage 3 - adaptation.** Either finetune the backbone and its decoder per
dataset for reconstruction, or attach a light task head to the pretrained
encoder and predict directly from projections with no reconstruction at
inference.

## Layout

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

## Data preparation

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
| `make_ctrate_patient_subsets.py` | nested patient-level CT-RATE training subsets for the limited-labelled-data experiment |

Dataset splits are not included: the eight evaluation datasets have their own
licences and access procedures.

## Running

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

`configs/stage2_v1_recononly.yaml` is the same Stage 2 with the semantic head
switched off, the ablation reported in the paper.

Reconstruction, one run per dataset and view count:

```bash
python finetune_foundation_recon.py --config configs/ours_head_converged_v8.yaml
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

Limited labelled data (the pretraining-benefit experiment): `reconeff_dual_n<N>`
finetunes the pretrained backbone and `reconeff_scratch_n<N>` trains the same
architecture from random initialisation (`from_scratch: true`), for `N` in
`{10, 25, 50, 100, 225, 450}` chest patients; the subsets come from
`preprocessing/make_ctrate_patient_subsets.py --sizes 10 25 50 100 225 450`, and
`recon_scratch_clssplit_converged_v8.yaml` is the all-patient random-init
counterpart.

Downstream probes attach a head to the pretrained encoder:

```bash
python train_repr_cls.py        --config configs/reprB_cls_pre_lp_v8.yaml   # linear probe
python train_repr_downstream.py --config configs/reprB_seg_pre_ft_v8.yaml   # finetune
```

## Requirements

Python 3.11 with PyTorch and CUDA, plus `nanodrr` for differentiable DRR
rendering. Tested with torch 2.5.1 / CUDA 12.4 on an NVIDIA GH200. See
`requirements.txt`.

## Notes on scope

This repository contains SPARC only. The baseline reconstructors we compare
against (DIF-Net, C2RV, DIF-Gaussian, SVCT, DeepSparse) are third-party code and
are not redistributed here; please obtain them from their original authors. Where
our modules build on ideas from those works - notably the projection-and-sample
lifting and the point decoder, which follow DeepSparse - the source is noted in
the file headers.

## License

Released under the Creative Commons Attribution-NonCommercial 4.0 International
License (CC BY-NC 4.0); see [LICENSE](LICENSE). Commercial use is not permitted
under this licence. Note that the evaluation datasets carry their own licences,
which this one does not override.
