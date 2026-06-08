# SLAM-Sandbox
With inspiration from the slam handbook, try to implement some sort of SLAM system.

## Dataset scaffold: EuROC MAV

- EuROC data mount point: `datasets/euroc_mav/`
- Notes and setup: `datasets/euroc_mav/README.md`

## EuROC visualization

Use the plotter on `state_groundtruth_estimate0/data.csv`:

```bash
python scripts/plot_euroc_trajectory.py /absolute/path/to/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv
```

Optional save-to-file:

```bash
python scripts/plot_euroc_trajectory.py /absolute/path/to/data.csv --out /tmp/trajectory.png
```

## More visual SLAM sources

See `resources/visual_slam_sources/README.md`.
