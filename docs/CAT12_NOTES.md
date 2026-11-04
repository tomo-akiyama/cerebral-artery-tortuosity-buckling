# CAT12 / SPM12 settings used in the study

Brain tissue segmentation and pial-surface reconstruction were performed with
**CAT12 (v12.9)** in **MATLAB R2023b (23.2)** using the CAT12 standard pipeline
on T1-weighted images that had already been resampled to 0.5 mm isotropic voxels.

Settings reported in the paper (GUI; not stored as a MATLAB batch in this repo):

- Bias correction strength: 0.75
- Preprocessing accuracy: 0.75
- Surface reconstruction: extra fine mesh (`vdist = 0.5`)

After CAT12, pial surfaces were isotropically remeshed with PyMeshLab
(`03_mesh/run_surface_remesh.py`): 10 iterations, target edge length 0.33 mm
(implemented as `targetlen=0.66` in voxel units with 0.5 mm voxels).

CAT12/SPM outputs are not shipped here. Users must install SPM12/CAT12 and run
the GUI pipeline locally after downloading IXI T1WI.
