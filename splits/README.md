# Data splits

The train, validation and test lists behind the paper's reconstruction results, one CSV per list, named
`<dataset>_{train,val,test}.csv` like the models and configs. Each file has one column, `ct_path`, giving a
volume as `processed/<dataset>/<case>.zarr` relative to `DATA_ROOT`. The file stem (`<case>`) is the case
identifier of the original dataset, so the lists can also be matched against the original downloads.
[`../preprocessing/README.md`](../preprocessing/README.md) says how each dataset was converted into these folders.

| Dataset | Files | Train | Validation | Test | Source volumes |
|---|---|---:|---:|---:|---|
| CT-RATE, official training split | `ctrate_pretrain.csv` | 47,149 | | | pretraining only (Stages 1 and 2) |
| CT-RATE, official validation split | `ctrate_{train,val,test}.csv` | 2,134 | 303 | 602 | 3,039 volumes, 1,304 patients |
| TotalSegmentator v2 | `totalsegmentator_{train,val,test}.csv` | 1,006 | 52 | 79 | 1,137 of 1,228 |
| Medical Segmentation Decathlon, six CT tasks | `msd_{train,val,test}.csv` | 986 | 212 | 209 | 1,410 |
| AbdomenCT-1K | `abdomenct1k_{train,val,test}.csv` | 744 | 159 | 158 | 1,062 |
| AMOS 2022, CT scans | `amos_{train,val,test}.csv` | 350 | 75 | 75 | 500 |
| CQ500 (head) | `cq500_{train,val,test}.csv` | 330 | 71 | 71 | 472 |
| ToothFairy3 (dental CBCT) | `toothfairy3_{train,val,test}.csv` | 372 | 80 | 77 | 532 |
| VerSe (spine) | `verse_{train,val,test}.csv` | 262 | 56 | 50 | 374 |

Counts are volumes, after the duplicate check below. CQ500, ToothFairy3 and VerSe are anatomies absent from pretraining.

## How each split was made

- **Pretraining.** `ctrate_pretrain.csv` is every volume of the CT-RATE official training split. Every chest
  reconstruction list comes from the official validation split, so no volume or patient used in pretraining
  appears in any evaluation list. The other seven datasets were not used in pretraining.
- **CT-RATE.** The official validation split was divided by patient: 913, 130 and 261 of its 1,304 patients
  (70/10/20), with all of a patient's volumes in one list. A volume `valid_<patient>_<scan>_<reconstruction>`
  belongs to patient `valid_<patient>`.
- **TotalSegmentator.** The dataset's own train/validation/test assignment (the `split` column of `meta.csv`,
  1,082/57/89 cases), restricted to the 1,137 cases whose reference segmentation in the dataset contains at
  least one of the liver, spleen, kidneys, aorta or lung lobes. The other 90 cases, and one volume we could not
  read (`s0589`), are in no list.
- **MSD, AbdomenCT-1K, AMOS and VerSe.** A random 70/15/15 split of all converted volumes, drawn once:

  ```python
  import random
  rows = sorted(paths)                  # every converted volume of the dataset
  random.Random(42).shuffle(rows)
  n = len(rows); k = round(0.15 * n)
  test, val, train = rows[:k], rows[k:2 * k], rows[2 * k:]
  ```
- **CQ500 and ToothFairy3.** The same 70/15/15 proportions with a different seed and order:

  ```python
  random.Random(0).shuffle(rows)
  train, val, test = rows[:n - 2 * k], rows[n - 2 * k:n - k], rows[n - k:]
  ```

These datasets give one case identifier per volume and, except VerSe, no separate patient identifier, so the
random splits are by volume. The duplicate check below found where that let one scan or subject into two lists.

## Duplicate check

Every test list was checked against its training and validation lists for repeated scans: by the SHA-1
digest of each voxel array, by the correlation of 32³ thumbnails (a pair counted as the same scan above 0.95,
against at most 0.88 between different CT-RATE patients), and by subject where the dataset identifies subjects.
Thirteen test volumes duplicated a training volume or came from a subject with one, and were removed from the
test lists: 3 in the Medical Segmentation Decathlon (its CT tasks share scans), 1 in AbdomenCT-1K, 3 in
ToothFairy3 and 6 in VerSe (a long spine scan can be split into several volumes of one subject). They are listed,
with the training volume each repeats, in [`excluded_test_volumes.csv`](excluded_test_volumes.csv). No model was
retrained; the paper's numbers use these reduced test lists. A few validation volumes still repeat a training
volume; validation only selects the checkpoint.

- **CT-RATE**: no patient is in more than one of the train, validation and test lists.
- **VerSe** identifies subjects (`sub-verse<n>`, `sub-gl<n>`). After the removals no test subject has a volume
  in the training or validation list.
- **The other datasets**: their test lists share no identical or near-identical scan with the training list.
