# Preprocessing

Entry point: `run_mri_preprocessing.py` (helpers in `preprocessing.py`).

Implements paper steps for resampling, MRA denoising/N4/skull-stripping, registration,
and hysteresis vessel segmentation. Paths inside the script historically point to a local
`data/raw` and `data/preprocessed` layout; adjust relative paths for your IXI download.
