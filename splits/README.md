# Data splits

The train, validation and test lists used in the paper, one CSV per split. Each file has one column,
`ct_path`, giving a volume as `processed/<folder>/<case>.zarr` relative to `DATA_ROOT`; the file stem
(`<case>`) is the case identifier of the original dataset, so the lists can also be matched against the
original downloads. The configs in `configs/` read these files from `$DATA_ROOT/splits/`.

| Dataset | Files | Train | Validation | Test | Converted volumes | Used for |
|---|---|---:|---:|---:|---|---|
| CT-RATE, official train split | `ct_rate_train_zarr.csv` | 47,149 | | | `processed/ct_rate_zarr/` | Stage 1 and Stage 2 pretraining |
| CT-RATE, official validation split | `cls3039_{train,val,test}_zarr.csv` | 2,134 | 303 | 602 | `processed/ct_rate_zarr_valid/` | chest reconstruction, classification, limited adaptation data |
| TotalSegmentator | `ts_seg_{train,val,test}.csv` | 1,006 | 52 | 79 | `processed/ts_all_zarr/` | reconstruction and organ segmentation |
| Medical Segmentation Decathlon, six CT tasks | `msd_full_{train,val,test}_zarr.csv` | 986 | 212 | 209 | `processed/msd_full_zarr/` | reconstruction |
| AbdomenCT-1K | `abdomenct1k_{train,val,test}_zarr.csv` | 744 | 159 | 158 | `processed/abdomenct1k_zarr/` | reconstruction |
| AMOS (CT) | `amos_ct_{train,val,test}_zarr.csv` | 350 | 75 | 75 | `processed/amos_ct_zarr/` | reconstruction |
| CQ500 (head) | `cq500_{train,val,test}_zarr.csv` | 330 | 71 | 71 | `processed/cq500_zarr/` | reconstruction, anatomy unseen in pretraining |
| ToothFairy3 (dental CBCT) | `toothfairy3_{train,val,test}_zarr.csv` | 372 | 80 | 77 | `processed/toothfairy3_zarr/` | reconstruction, anatomy unseen in pretraining |
| VerSe (spine) | `verse_{train,val,test}_zarr.csv` | 262 | 56 | 50 | `processed/verse_zarr/` | reconstruction, anatomy unseen in pretraining |

Counts are volumes, after the duplicate check below.

**Duplicate check.** Every test list was checked against its training and validation lists for repeated scans:
by the SHA-1 digest of each voxel array, by the correlation of 32³ thumbnails (a pair counted as the same scan
above 0.95, against at most 0.88 between different CT-RATE patients), and by subject where the dataset identifies
subjects. Thirteen test volumes duplicated a training volume or came from a subject with one, and were removed
from the test lists: 3 in the Medical Segmentation Decathlon (its CT tasks share scans), 1 in AbdomenCT-1K, 3 in
ToothFairy3 and 6 in VerSe (a long spine scan can be split into several volumes of one subject). They are listed,
with the training volume each repeats, in [`excluded_test_volumes.csv`](excluded_test_volumes.csv). No model was
retrained; the paper's numbers use these reduced test lists. A few validation volumes still repeat a training
volume; validation only selects the checkpoint.

- **CT-RATE** has several volumes per patient; a volume `valid_<patient>_<scan>_<reconstruction>` belongs to
  patient `valid_<patient>`. The train, validation and test lists hold 913, 130 and 261 patients, and no patient
  is in more than one of them.
- **VerSe** identifies subjects (`sub-verse<n>`, `sub-gl<n>`). After the removals no test subject has a volume in
  the training or validation list.
- **The other datasets** give one case identifier per volume and no separate patient identifier; their test lists
  share no identical or near-identical scan with the training list.

The pretraining list is the whole CT-RATE official training split; every evaluation list for the chest
comes from the CT-RATE official validation split, so no volume or patient used in pretraining appears in
any chest train, validation or test list. The other eight datasets were not used in pretraining.

## Limited adaptation data

Nested subsets of the CT-RATE training list, drawn once by patient with a fixed seed
(`preprocessing/make_ctrate_patient_subsets.py`), so every method sees the same patients at every budget.
Validation and test lists are unchanged.

| File | Patients | Volumes |
|---|---:|---:|
| `cls3039_train_pat10.csv` | 10 | 28 |
| `cls3039_train_pat25.csv` | 25 | 64 |
| `cls3039_train_pat50.csv` | 50 | 127 |
| `cls3039_train_pat100.csv` | 100 | 252 |
| `cls3039_train_pat225.csv` | 225 | 531 |
| `cls3039_train_pat450.csv` | 450 | 1,079 |
