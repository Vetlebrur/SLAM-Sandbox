# EuROC MAV dataset

This folder is a local dataset mount point for EuROC MAV data.

## Option A: put extracted sequences directly here

Example layout:

- `datasets/euroc_mav/MH_01_easy/`
- `datasets/euroc_mav/MH_02_easy/`

## Option B: track a separate dataset repo as a submodule-like source

If you keep EuROC data in another Git repository, you can attach it here:

```bash
git submodule add <your-euroc-data-repo-url> datasets/euroc_mav/data
```

> Note: EuROC is large; many teams keep data outside the main code repo and mount/symlink it here.

## Source

Official dataset page: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
