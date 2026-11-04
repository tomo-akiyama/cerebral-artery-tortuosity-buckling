# Manual vessel segmentation QC (not automated)

After hysteresis thresholding (`01_preprocessing/run_mri_preprocessing.py`), vessel masks
were inspected and edited in **3D Slicer (v5.6)** by a clinical expert.

Typical manual steps (cannot be fully coded):

1. Load the coregistered, skull-stripped MRA and the automatic vessel mask.
2. Remove misclassified voxels and artificial bridges between adjacent arteries
   or between arteries and nearby high-signal structures (e.g., veins).
3. Remove perforating arteries branching from major trunks when they are not
   part of the analysis tree.
4. Restore anatomical continuity only when a disconnection is clearly spurious
   on the original MRA.
5. Isolate the middle cerebral arteries (MCAs) for analysis (accessory/duplicated
   MCA variants were retained when present).

Inlets/outlets for VMTK centerline extraction were placed as 3D Slicer markups
(`*.mrk.json`). Raw images, masks, meshes, and markups are **not** included in
this public repository (IXI download + local preprocessing required to regenerate).
