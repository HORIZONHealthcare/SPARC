# Per-volume results

`reconstruction/<dataset>_V<views>_<method>.csv` gives, for every test volume, the PSNR (`psnr_db`) and SSIM
(`ssim`) of one method's reconstruction, computed over [−1,024, 1,024] HU normalised to [0, 1]. `ct_path` is the
case identifier used in [`../splits/`](../splits/). The rows are the test lists used in the paper, after the duplicate
check described in [`../splits/README.md`](../splits/README.md).

| Field | Values |
|---|---|
| `<dataset>` | `ct_rate`, `totalseg`, `msd`, `abdomenct1k`, `amos`, `cq500`, `toothfairy3`, `verse` |
| `<views>` | `4`, `8` (each view count is a separately trained model) |
| `<method>` | `ours` (SPARC), `ds` (DeepSparse), `svct` (SVCT), `c2rv` (C²RV), `difgs` (DIF-Gaussian), `difnet` (DIF-Net) |

Every method was retrained under the same data, geometry and training protocol. The paper's reconstruction tables
report the column means; its confidence intervals come from a bootstrap over patients (10,000 resamples). CT-RATE has
several volumes per patient (`valid_<patient>_<scan>_<reconstruction>`), which the bootstrap resamples together.
SPARC's rows can be reproduced from the released weights with the command in the main README.
