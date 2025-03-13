"""This unittest was written with the help of ChatGPT"""

import unittest
import numpy as np
import xarray as xr
from optim_esm_tools.analyze.discontinuous_grid_patcher import DiscontinuousGridPatcher


class TestDiscontinuousGridPatcher(unittest.TestCase):
    def setUp(self):
        """Set up a synthetic dataset with missing values to test the patching."""
        time = np.arange(3)  # Multiple time steps
        lat = np.linspace(-89.5, 89.5, 180)
        lon = np.linspace(-179.5, 179.5, 360)
        data = np.random.rand(len(time), len(lat), len(lon))
        data[:, :, 50:52] = np.nan  # Introduce missing values at a specific longitude

        cell_area = np.ones((len(lat), len(lon)))  # Dummy cell area values

        self.ds = xr.Dataset(
            {
                "variable": (['time', 'lat', 'lon'], data),
                "cell_area": (['lat', 'lon'], cell_area),
            },
            coords={"time": time, "lat": lat, "lon": lon},
            attrs={"variable_id": "variable"},  # Set the variable_id attribute
        )
        self.should_have_data_mask = xr.DataArray(
            np.ones((len(lat), len(lon)), dtype=bool),  # Expect data everywhere
            coords={"lat": lat, "lon": lon},
            dims=["lat", "lon"],
        )
        self.patcher = DiscontinuousGridPatcher(self.ds, self.should_have_data_mask)

    def test_find_issues(self):
        """Test if the patcher correctly identifies the missing data as an issue."""
        issues = self.patcher.find_issues()
        self.assertTrue(len(issues) > 0, "No issues detected when they should be.")

    def test_patch_all_issues(self):
        """Test if the patching mechanism correctly fills in missing values."""
        patched_ds = self.patcher.patch_all_issues()
        self.assertFalse(
            np.isnan(patched_ds["variable"].values).any(),
            "Patching did not fill all missing values.",
        )

    def test_patch_all_issues_with_iter_time(self):
        """Test patching when iter_time is enabled."""
        patcher = DiscontinuousGridPatcher(
            self.ds,
            self.should_have_data_mask,
            iter_time=True,
        )
        patched_ds = patcher.patch_all_issues()
        self.assertFalse(
            np.isnan(patched_ds["variable"].values).any(),
            "Patching with iter_time did not fill all missing values.",
        )

    def test_invalid_dataset_dimension(self):
        """Test if an incorrect dataset dimensionality raises NotImplementedError when executing patch."""

        alt_data = np.random.rand(3, 2, 4, 5)  # Incorrect dimensions
        alt_ds = xr.Dataset(
            {"variable": (['a', 'b', 'lat', 'lon'], alt_data)},
            coords={
                "a": np.arange(3),
                "b": np.arange(2),
                "lat": np.arange(4),
                "lon": np.arange(5),
            },
        )
        alt_ds['variable'].data[:, :, :, 2:5] = np.nan
        patcher = DiscontinuousGridPatcher(
            alt_ds,
            xr.ones_like(alt_ds['variable'].isel(a=0)),
        )

        with self.assertRaises(NotImplementedError):
            patcher.execute_patch(np.ones((4, 5), dtype=bool))


if __name__ == "__main__":
    unittest.main()
