# HMI Solar Wind Forecasting Project

This project builds HMI-magnetogram-derived coronal-hole features and evaluates whether they improve XGBoost solar wind speed forecasting.

## Current Structure

```text
hmi/
├── README.md
├── PROJECT_SUMMARY.md
├── data/
│   ├── 4x3.csv
│   ├── renamed_no_proxy.csv
│   ├── hmi_*_6h*.csv
│   ├── run_hmi_year.py
│   ├── solar_wind_boosting_pipeline.ipynb
│   ├── solar_wind_boosting_pipeline_colab.ipynb
│   ├── stageB_cross_regime_*.csv
│   └── enhancements/
├── logs/
│   ├── nbconvert_*.log
│   └── run_*.log
├── presentation/
│   └── stage_AB_summary.tex
├── scripts/
│   └── experiments/
├── transfer/
│   ├── cross_validation.py
│   ├── preprocessing_dataset.py
│   └── test.ipynb
└── archive/
    ├── root_legacy/
    ├── scratch/
    └── presentation_stale_figures/
```

## Notes

- The canonical modeling data and notebooks remain in `data/` because the notebooks load files by local filename when run from that directory.
- `logs/` contains execution logs moved out of `data/`.
- `archive/root_legacy/` contains stale root-level copies that were already marked as superseded in `PROJECT_SUMMARY.md`.
- `archive/scratch/` contains one-off notebook/script artifacts from previous runs.
- `archive/presentation_stale_figures/` contains old figures that the current TeX deck no longer references.

## Common Entry Points

- HMI feature generation: `data/run_hmi_year.py`
- Main local notebook: `data/solar_wind_boosting_pipeline.ipynb`
- Colab notebook: `data/solar_wind_boosting_pipeline_colab.ipynb`
- Long-form handoff: `PROJECT_SUMMARY.md`
