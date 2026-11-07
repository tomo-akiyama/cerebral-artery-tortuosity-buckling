# Environment

VMTK requires an x86_64 (Intel / AMD) Conda environment.

- Windows / Linux with Intel or AMD CPUs are supported.
- On Apple Silicon (arm64) Macs, VMTK must be run under Rosetta using an x86_64 Conda distribution.

```bash
conda env create -f environment_vmtk_intel.yml
conda activate vmtk_intel
```

See `environment_vmtk_intel.yml` (and the companion pin files in this folder) for details of the Python/Conda environment used in this study.

The repository root `environment.yml` is the minimal publication recipe.
