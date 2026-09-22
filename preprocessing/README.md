# Preprocessing

The training and evaluation code reads every CT volume as a zarr array made by the scripts in this folder.
This page gives, for each dataset in the paper, the files we converted and the commands. Each command writes
to `$DATA_ROOT/processed/<dataset>/`, the folder the lists in [`../splits/`](../splits/) point to, and writes a
manifest of every converted volume (`--out-manifest`); the paper's split lists are subsets of these manifests.

## What every converter does

| Step | Detail |
|---|---|
| Intensities | Hounsfield units, clipped to [-1024, 3071] and stored as int16. CT-RATE needs a rescale from its metadata (below); the other datasets store HU already. The models clip further to [-1024, 1024] when loading (`hu_min` / `hu_max` in each config). |
| Geometry | The source voxel grid is kept: no resampling, cropping or reorientation. Arrays keep the source voxel order, (x, y, z) with z the slice axis, and the voxel spacing in mm is stored with them. |
| Attributes | `spacing` (mm, x/y/z), `bbox_lo` / `bbox_hi` (voxel bounding box of HU > -500, the body), `shape`, `volume_name`, `done` |
| Storage | zarr v2, 64³ chunks, Blosc zstd level 3; one `<case>.zarr` per volume, named after the source case |
| Restarts | A volume already marked `done` is skipped, so an interrupted run can be restarted |

Resampling happens later, in the data loader: each config sets the voxel spacing of the 256³ grid that the
projections are rendered from and the reconstruction is compared against (at test time 1.6 mm for chest,
abdomen and spine, 1.2 mm for the head and 0.625 mm for dental CBCT), together with the acquisition geometry.

Every converter takes a CSV manifest with one column, `ct_path`, except the DICOM converter, which searches a
folder. Paths may be absolute or relative to `DATA_ROOT`. The commands below build each manifest with `find`;
`! -name '._*'` skips macOS resource files.

| Dataset | Files converted | Converted | Script | Output folder |
|---|---|---:|---|---|
| CT-RATE | `dataset/train/**/*.nii.gz`, `dataset/valid/**/*.nii.gz` | 47,149 + 3,039 | `preprocess_ctrate_zarr.py` | `processed/ctrate/` |
| TotalSegmentator v2 | `s*/ct.nii.gz` | 1,227 of 1,228 | `preprocess_ts_zarr.py` | `processed/totalsegmentator/` |
| Medical Segmentation Decathlon | `imagesTr`, `imagesTs` of Tasks 03, 06, 07, 08, 09, 10 | 1,410 | `preprocess_nii_zarr.py` | `processed/msd/` |
| AbdomenCT-1K | `AbdomenCT-1K-ImagePart{1,2,3}/*.nii.gz` | 1,062 | `preprocess_nii_zarr.py` | `processed/abdomenct1k/` |
| AMOS 2022 | `images{Tr,Va,Ts}/amos_0001` ... `amos_0500` (CT) | 500 | `preprocess_nii_zarr.py` | `processed/amos/` |
| CQ500 | DICOM studies `CQ500CT*` | 472 | `preprocess_dicom_zarr.py` | `processed/cq500/` |
| ToothFairy3 | `imagesTr/*.nii.gz` | 532 | `preprocess_nii_zarr.py` | `processed/toothfairy3/` |
| VerSe (CADS copy) | `0010_verse/images/*.nii.gz` | 374 | `preprocess_nii_zarr.py` | `processed/verse/` |

In the commands, `/path/to/...` is where you downloaded each dataset.

## CT-RATE

From [Hugging Face](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE): the official training split
(47,149 volumes, used for pretraining) and validation split (3,039 volumes, used for chest reconstruction).
CT-RATE's NIfTI files hold raw stored values and an identity affine, so `preprocess_ctrate_zarr.py` recovers
Hounsfield units (`stored * RescaleSlope + RescaleIntercept`) and the voxel spacing (`XYSpacing`, `ZSpacing`)
from the metadata CSV of each split. Both splits go into one folder; their case names (`train_*`, `valid_*`)
cannot collide.

```bash
CTRATE=/path/to/CT-RATE/dataset
for split in train valid; do
  meta=$CTRATE/metadata/$([ $split = train ] && echo train_metadata.csv || echo validation_metadata.csv)
  { echo ct_path; find $CTRATE/$split -name '*.nii.gz' ! -name '._*' | sort; } > ctrate_${split}_nii.csv
  python preprocessing/preprocess_ctrate_zarr.py --manifest ctrate_${split}_nii.csv --metadata $meta \
      --out-dir $DATA_ROOT/processed/ctrate --out-manifest ctrate_${split}_all.csv
done
```

## TotalSegmentator

TotalSegmentator v2 (1,228 CT volumes, [Zenodo](https://doi.org/10.5281/zenodo.6802613)), one folder per case
with the CT in `ct.nii.gz`. The files store HU and real spacing. Each zarr is named after its case folder
(`s0000.zarr`). One volume in our copy, `s0589`, could not be read. Which of the rest are in the splits is
described in [`../splits/README.md`](../splits/README.md).

```bash
{ echo ct_path; find /path/to/Totalsegmentator_dataset -name ct.nii.gz | sort; } > totalsegmentator_nii.csv
python preprocessing/preprocess_ts_zarr.py --manifest totalsegmentator_nii.csv \
    --out-dir $DATA_ROOT/processed/totalsegmentator --out-manifest totalsegmentator_all.csv
```

## Medical Segmentation Decathlon

The six CT tasks (Task03 Liver, Task06 Lung, Task07 Pancreas, Task08 Hepatic Vessel, Task09 Spleen,
Task10 Colon), both `imagesTr` and `imagesTs`, since reconstruction needs no labels: 1,410 volumes. The case
names carry the task (`liver_0`, `lung_001`, ...), so they are unique across tasks. Some tasks share scans;
test volumes that repeat a training scan were removed from the test list (see the splits page).

```bash
MSD=/path/to/MSD
{ echo ct_path
  for t in Task03_Liver Task06_Lung Task07_Pancreas Task08_HepaticVessel Task09_Spleen Task10_Colon; do
    find $MSD/$t/imagesTr $MSD/$t/imagesTs -name '*.nii.gz' ! -name '._*'
  done | sort; } > msd_nii.csv
python preprocessing/preprocess_nii_zarr.py --manifest msd_nii.csv \
    --out-dir $DATA_ROOT/processed/msd --out-manifest msd_all.csv
```

## AbdomenCT-1K

All three image parts, 1,062 volumes (`Case_00001_0000`, ...).

```bash
{ echo ct_path; find /path/to/AbdomenCT-1K/AbdomenCT-1K-ImagePart{1,2,3} -name '*.nii.gz' ! -name '._*' | sort; } > abdomenct1k_nii.csv
python preprocessing/preprocess_nii_zarr.py --manifest abdomenct1k_nii.csv \
    --out-dir $DATA_ROOT/processed/abdomenct1k --out-manifest abdomenct1k_all.csv
```

## AMOS 2022

The CT scans only: `amos_0001` to `amos_0500` from `imagesTr`, `imagesVa` and `imagesTs` (500 volumes).
Higher numbers are MRI and were not used.

```bash
{ echo ct_path; find /path/to/amos22/imagesTr /path/to/amos22/imagesVa /path/to/amos22/imagesTs -name 'amos_*.nii.gz' \
    | grep -E 'amos_(0[0-4][0-9]{2}|0500)\.nii\.gz$' | sort; } > amos_nii.csv
python preprocessing/preprocess_nii_zarr.py --manifest amos_nii.csv \
    --out-dir $DATA_ROOT/processed/amos --out-manifest amos_all.csv
```

## CQ500

Head CT in DICOM. We used the [Kaggle mirror](https://www.kaggle.com/datasets/crawford/qureai-headct) of the
[qure.ai release](http://headctstudy.qure.ai/dataset). `preprocess_dicom_zarr.py` finds the study folders
(`CQ500CT<n> ...`), takes from each study the series folder with the most slices (at least 20, which skips
scouts), and reads it with SimpleITK, which applies the DICOM rescale to give HU. Each zarr is named after the
study (`CQ500CT0.zarr`). The archive has 492 study folders for 473 studies, because 18 studies appear twice;
the first copy is kept. One study, `CQ500CT163`, could not be converted, leaving 472.

```bash
python preprocessing/preprocess_dicom_zarr.py --root /path/to/qureai-headct --study-glob 'CQ500CT*' \
    --out-dir $DATA_ROOT/processed/cq500 --out-manifest cq500_all.csv
```

## ToothFairy3

Dental cone-beam CT from the [ToothFairy3 release](https://toothfairy3.grand-challenge.org/dataset/) (registration
required): the 532 volumes in `imagesTr` (63 `ToothFairy3F`, 417 `ToothFairy3P`, 52 `ToothFairy3S`), at 0.3 mm.
CBCT values are not calibrated Hounsfield units; they are used as released (air near -1000) and clipped like CT.

```bash
{ echo ct_path; find /path/to/ToothFairy3/imagesTr -name '*.nii.gz' ! -name '._*' | sort; } > toothfairy3_nii.csv
python preprocessing/preprocess_nii_zarr.py --manifest toothfairy3_nii.csv \
    --out-dir $DATA_ROOT/processed/toothfairy3 --out-manifest toothfairy3_all.csv
```

## VerSe

The 374 CT scans of VerSe'19 and VerSe'20. We used the copy in the
[CADS collection](https://huggingface.co/datasets/huggingface/CADS-dataset) (folder `0010_verse/images`), whose
files keep the VerSe subject names with a `_0000` suffix (`sub-verse402_split-verse202_ct_0000`). The original
release is on OSF ([VerSe'19](https://osf.io/nqjyw/), [VerSe'20](https://osf.io/t98fz/)).

```bash
hf download huggingface/CADS-dataset --repo-type dataset --include "0010_verse/images/*" --local-dir /path/to/CADS
{ echo ct_path; find /path/to/CADS/0010_verse/images -name '*.nii.gz' ! -name '._*' | sort; } > verse_nii.csv
python preprocessing/preprocess_nii_zarr.py --manifest verse_nii.csv \
    --out-dir $DATA_ROOT/processed/verse --out-manifest verse_all.csv
```
