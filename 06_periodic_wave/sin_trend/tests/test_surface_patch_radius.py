"""A changed patch radius must reach mesh points beyond the former crop box."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sin_trend_measure import default_measurement_config
from sin_trend_surface import run_surface_evaluation, surface_settings


class TestSurfacePatchRadius(unittest.TestCase):
    def test_large_patch_reaches_plane_outside_original_five_mm_crop(self):
        try:
            import pyvista as pv
        except ImportError:
            self.skipTest("pyvista is required for the real mesh test")
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = Path(tmp) / "plane.vtp"
            # Coordinates are voxels: the plane is 6 mm above the centerline.
            pv.Plane(center=(0, 0, 12), direction=(0, 0, 1),
                     i_size=60, j_size=40, i_resolution=100, j_resolution=70).save(mesh_path)
            x = np.linspace(-10, 10, 101)
            P = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
            d = np.column_stack((np.zeros_like(x), np.cos(x), np.zeros_like(x)))
            settings = surface_settings(default_measurement_config())
            ev, est, _ = run_surface_evaluation(
                P_full_vox=P, P_trim=P, d_sin=d, ds_vox=.2,
                r_patch_vox=14, mesh_path=mesh_path, settings=settings,
            )
            self.assertEqual(ev["status"], "ok")
            self.assertGreater(ev["constraint_coverage"], .9)
            self.assertAlmostEqual(ev["R_out"], 0, places=12)
            self.assertTrue(np.asarray(est["mask_eval"]).any())


if __name__ == "__main__":
    unittest.main()
