# vmtk_intel environment snapshot (osx-64)

**This folder is NOT the paper publication recipe.**

| Location | Purpose |
|----------|---------|
| Repository root `environment.yml` | Minimal dependency list for publication / methods |
| **This folder** | Exact working snapshot of the analysis conda env `vmtk_intel` on macOS **osx-64** (Intel / Rosetta) |

Exported from machine: Apple Silicon Mac running `vmtk_intel` as **x86_64** under Rosetta.  
Date context: 2026-10-07.

## Files

| File | Role |
|------|------|
| `environment_vmtk_intel_explicit.txt` | Conda packages pinned by URL (`conda create --file`) |
| `requirements_vmtk_intel_pypi_only.txt` | Pip-only packages (channel=pypi in conda list) |
| `environment_vmtk_intel.yml` | Combined conda+pip export (convenient, slightly less strict on builds) |
| `requirements_vmtk_intel_pip.txt` | Full `pip freeze` (includes conda-visible pkgs; **do not** use alone after explicit create) |

## Recreate on another osx-64 Mac

```bash
conda create -n vmtk_intel --file environment_vmtk_intel_explicit.txt
conda activate vmtk_intel
pip install -r requirements_vmtk_intel_pypi_only.txt
```

Or (less strict on conda builds, includes pip in one step):

```bash
conda env create -f environment_vmtk_intel.yml
```

Do **not** confuse with root `environment.yml`.
