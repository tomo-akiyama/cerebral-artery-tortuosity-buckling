"""PyMeshLab isotropic remesh helpers for brain surfaces."""
import os
import pymeshlab

import mesh_generation


def remesh_ply(
    input_ply_path,
    output_ply_path,
    iterations=10,
    targetlen=0.66,
    featuredeg=30,
    maxsurfdist_percent=1,
):
    """
    Apply pymeshlab isotropic explicit remeshing to a PLY mesh.

    Notes
    -----
    Mesh coordinates are in voxel units with voxel_size = 0.5 mm, so
    targetlen=0.66 corresponds to ~0.33 mm edge length.
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_ply_path)
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations,
        targetlen=pymeshlab.PureValue(targetlen),
        featuredeg=featuredeg,
        maxsurfdist=pymeshlab.PercentageValue(maxsurfdist_percent),
    )
    ms.save_current_mesh(output_ply_path)
    return output_ply_path


def convert_gifti_pial_to_ply(gii_path, nii_path, output_ply_path):
    """Convert CAT12 GIFTI pial surface to PLY via mesh_generation.gifti_to_vtkpolydata."""
    polydata = mesh_generation.gifti_to_vtkpolydata(gii_path, nii_path)
    mesh_generation.export_vtkpolydata_to_ply(polydata, output_ply_path)
    return output_ply_path


def process_hemisphere(
    side,
    gii_path,
    nii_path,
    output_dir,
    directory_id,
    remesh_kwargs=None,
):
    """
    Convert one hemisphere pial GIFTI to PLY and remesh.

    Writes:
      {directory_id}_{side}_pial.ply
      {directory_id}_{side}_pial_remesh.ply
    where side is 'lt' or 'rt'.
    """
    if remesh_kwargs is None:
        remesh_kwargs = {}

    pial_ply_path = os.path.join(output_dir, f"{directory_id}_{side}_pial.ply")
    remesh_ply_path = os.path.join(output_dir, f"{directory_id}_{side}_pial_remesh.ply")

    print(f"Converting {side} pial GIFTI → PLY...")
    convert_gifti_pial_to_ply(gii_path, nii_path, pial_ply_path)
    print(f"  Saved: {pial_ply_path}")

    print(f"Remeshing {side} pial...")
    remesh_ply(pial_ply_path, remesh_ply_path, **remesh_kwargs)
    print(f"  Saved: {remesh_ply_path}")

    return pial_ply_path, remesh_ply_path
