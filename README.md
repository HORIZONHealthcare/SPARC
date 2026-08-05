# SPARC

Reference implementation of **SPARC**, a geometry-aware foundation model for
sparse-view CT reconstruction: a handful of X-ray projections in, a full 3D
volume out, with an encoder whose features also transfer to downstream
recognition without reconstructing anything at inference.

> Y. Lin, J. Xu, L. Ju, H. Wang, J. Guo, H. Yao, H. Sun, Y. Zhou.
> *A Geometry-Aware Foundation Model for Sparse-View CT Reconstruction.*

## Method

Three stages:

**Stage 1 — CT semantic target.** A 3D masked autoencoder (ViT-L/16, 0.75 mask
ratio) is trained on CT volumes alone. Its decoder is discarded and its encoder
frozen; it maps a CT to `16³ = 4096` anatomy-aware tokens that serve as the
prediction target in Stage 2.

**Stage 2 — sparse-view backbone.** Projections are rendered online with
randomised acquisition geometry (view count `V ∈ [2,16]`; uniform-circular,
random-circular, limited-angle, bi-planar and clustered patterns; randomised
source/detector distances). Each view passes through a shared ConvNeXt pyramid
with a transformer at the coarsest scale, into which the 6-D Plücker coordinate
of every ray is injected, so the encoder reasons about *where* a projection was
taken rather than treating views as an unordered set. A `16³` grid of points is
projected into all views, bilinearly sampled, max-fused across views and refined
by a 3D transformer into the feature volume `F ∈ R^{16³ × 768}`. Two heads
consume `F`: an implicit point decoder supervised by voxel reconstruction
(primary), and a light head that predicts the frozen CT-MAE tokens under a
smooth-L1 cross-modal joint-embedding loss (auxiliary, weight `λ = 1`). The
semantic head is discarded afterwards.

**Stage 3 — adaptation.** Either finetune the backbone and its decoder per
dataset for reconstruction, or attach a light task head to the pretrained
encoder and predict directly from projections with no reconstruction at
inference.

## Layout

```
src/models/foundation.py         SparseViewFoundation: one backbone, two heads
src/models/xray_encoder_2d.py    per-view ConvNeXt pyramid + Plücker injection
src/models/feature_volume_3d.py  projection, max-fuse, 3D transformer -> F
src/models/point_decoder.py      implicit point decoder (resolution-free)
src/models/ct_mae.py             Stage-1 3D masked autoencoder
src/utils/plucker.py             Plücker ray maths
src/utils/train_common.py        epoch budget, early stopping, exact resume

train_ctmae.py                   Stage 1
train.py                         Stage 2
finetune_foundation_recon.py     Stage 3, reconstruction
train_repr_cls.py                Stage 3, classification head
train_repr_downstream.py         Stage 3, segmentation head
```

## Running

Configs reference data through two placeholders; point them at your own layout:

```bash
export DATA_ROOT=/path/to/data        # expects $DATA_ROOT/splits/*.csv manifests
export OUTPUT_ROOT=/path/to/outputs
```

A manifest is a CSV listing one volume path per row. Then:

```bash
python train_ctmae.py               --config configs/ctmae_pretrain_crop.yaml
python train.py                     --config configs/stage2_foundation.yaml
python finetune_foundation_recon.py --config configs/recon_ours_clssplit_converged_v8.yaml
```

Every reconstruction config runs a fixed protocol: at most 400 epochs, early
stopping on held-out validation PSNR (minimum 120 epochs, then five consecutive
validations without a 0.02 dB improvement), validating every 20 epochs, and the
best-validation checkpoint is the one evaluated on test.

Downstream probes attach a head to the pretrained encoder:

```bash
python train_repr_cls.py        --config configs/reprB_cls_pre_lp_v8.yaml   # linear probe
python train_repr_downstream.py --config configs/reprB_seg_pre_ft_v8.yaml   # finetune
```

## Requirements

PyTorch with CUDA, plus `nanodrr` for differentiable DRR rendering. See
`requirements.txt`.

## Notes on scope

This repository contains SPARC only. The baseline reconstructors we compare
against (DIF-Net, C²RV, DIF-Gaussian, SVCT, DeepSparse) are third-party code and
are not redistributed here; please obtain them from their original authors. Where
our modules build on ideas from those works — notably the projection-and-sample
lifting and the point decoder, which follow DeepSparse — the source is noted in
the file headers.

Dataset splits are not included: the eight evaluation datasets have their own
licences and access procedures.

## License

Released under the Creative Commons Attribution-NonCommercial 4.0 International
License (CC BY-NC 4.0); see [LICENSE](LICENSE). Commercial use is not permitted
under this licence. Note that the evaluation datasets carry their own licences,
which this one does not override.
