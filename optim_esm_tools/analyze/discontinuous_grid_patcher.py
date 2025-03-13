import typing as ty

import numpy as np
import optim_esm_tools as oet
import xarray as xr
from statsmodels.stats.weightstats import DescrStatsW


class DiscontinuousGridPatcher:
    min_samples_for_issue: int = 75
    max_lon_weighted_std: ty.Union[float, int] = 4  # degrees, longitude

    def __init__(
        self,
        ds: xr.Dataset,
        should_have_data_mask: xr.DataArray,
        iter_time: bool = False,
        build_cluster_kw: ty.Optional[dict] = None,
        split_cluster_kw: ty.Optional[dict] = None,
    ):
        """Patch holes in regridded data to allow continuous region finding.

        Args:
            ds (xr.Dataset): dataset to patch
            should_have_data_mask (xr.DataArray): a lat/lon mask of where there should be data
            iter_time (bool, optional): Iteratate over the time field. Defaults to False.
            build_cluster_kw (ty.Optional[dict], optional): Optional arguments passed to
                optim_esm_tools.analyze.clustering.build_cluster_mask. Defaults to None.
            split_cluster_kw (ty.Optional[dict], optional): Optional arguments passed to
            optim_esm_tools.analyze.clustering._split_to_continuous. Defaults to None.
        """
        self.ds = ds.copy()
        self.should_have_data_mask: xr.DataArray = should_have_data_mask
        self.iter_time: bool = iter_time
        self.build_cluster_kw: ty.Dict = build_cluster_kw or dict(
            min_samples=4,
            max_distance_km=120,
        )
        self.split_cluster_kw: ty.Dict = split_cluster_kw or dict(
            add_diagonal=False,
            add_double_lon=False,
        )

    def find_issues(self) -> ty.List[np.ndarray]:
        """Find issues in the dataset, if any return a list of masks where
        issues exist.

        Returns:
            ty.List[np.ndarray]: List of issues
        """
        year_0 = self.ds[self.ds.variable_id].isel(time=0)
        res = oet.analyze.clustering.build_cluster_mask(
            (self.should_have_data_mask & year_0.isnull()).values.astype(bool),
            year_0.lat.values,
            year_0.lon.values,
            **self.build_cluster_kw,
        )
        _, lon = np.meshgrid(self.ds.lat.values, self.ds.lon.values)
        cell_a = self.ds["cell_area"].values

        continous_masks = oet.analyze.clustering._split_to_continuous(
            res[0],
            **self.split_cluster_kw,
        )
        large_enough_masks = [
            r for r in continous_masks if r.sum() >= self.min_samples_for_issue
        ]

        located_enough_masks = [
            m
            for m in large_enough_masks
            if DescrStatsW(lon.T[m], weights=cell_a[m], ddof=0).std
            < self.max_lon_weighted_std
        ]
        if len(continous_masks):
            oet.get_logger().info(
                f"Started with {len(continous_masks)} of which {len(large_enough_masks)} are >= {self.min_samples_for_issue} cells large. "
                f"Finally, {len(located_enough_masks)} are actually also located enough (std < {self.max_lon_weighted_std}).",
            )
        return located_enough_masks

    @staticmethod
    def find_adjacency(mask: np.ndarray):
        """Get cells that are one around the mask."""
        import scipy.ndimage as ndimage

        blurred = (
            ndimage.gaussian_filter(
                (mask > 0).astype(np.float64),
                sigma=(0, 1),
                order=0,
            )
            > 0.2
        ).astype(np.int16)

        return (blurred - mask.astype(np.int16)).astype(np.bool_)

    def execute_patches(self, issues: ty.List[np.ndarray]) -> None:
        for issue in issues:
            self.execute_patch(issue)

    def patch_one_variable(
        self,
        variable: str,
        issue: np.ndarray,
        adjacency_mask: np.ndarray,
        iter_time: ty.Optional[bool] = None,
    ) -> None:
        """Patch the data in the dataset stored under "variable".

        There are only a limited number of datastructures supported, if the NotImplementedError is raised,
        one should add the appropriate handling proceidure for the datastructure.

        Args:
            variable (str): the data-array in the dataset to fix
            issue (np.ndarray): the 2d-boolean mask for the issue to fix
            adjacency_mask (np.ndarray): the result of self.find_adjacency(issue)
            iter_time (ty.Optional[bool], optional): Iteratate over the time field. Defaults to None.

        Raises:
            NotImplementedError: If the data structure is unrecognized
        """
        iter_time = iter_time if iter_time is not None else self.iter_time
        da = self.ds[variable].copy()
        del self.ds[variable]
        idx_lon = np.argwhere(np.array(da.dims) == "lon").squeeze()
        idx_lat = np.argwhere(np.array(da.dims) == "lat").squeeze()

        if idx_lat == 0 and idx_lon == 1 and len(da.shape) == 2:
            values = da.values
            temp = values.copy()
            temp[~adjacency_mask] = np.nan
            patch_data = np.nanmean(temp, axis=idx_lon)
            values[issue] = np.repeat(patch_data, values.shape[idx_lon]).reshape(
                values.shape,
            )[issue]

            da.data = values
        elif idx_lat == 1 and idx_lon == 2 and len(da.shape) == 3 and not iter_time:
            values = da.values
            temp = values.copy()
            temp[:, ~adjacency_mask] = np.nan
            patch_data = np.nanmean(temp, axis=idx_lon)
            values[:, issue] = np.repeat(patch_data, values.shape[idx_lon]).reshape(
                values.shape,
            )[:, issue]

            da.data = values
        elif idx_lat == 1 and idx_lon == 2 and len(da.shape) == 3 and iter_time:
            for t_idx in oet.utils.tqdm(list(range(len(self.ds["time"])))):
                da_t = da.isel(time=t_idx)
                values = da_t.values
                temp = values.copy()
                temp[~adjacency_mask] = np.nan
                patch_data = np.nanmean(temp, axis=idx_lon - 1)
                values[issue] = np.repeat(
                    patch_data,
                    values.shape[idx_lon - 1],
                ).reshape(values.shape)[issue]

                da.data[t_idx] = values

        else:
            raise NotImplementedError(
                idx_lat == 1,
                idx_lon == 2,
                len(da.shape) == 3,
                not iter_time,
            )

        self.ds[variable] = da

    def execute_patch(self, issue: np.ndarray) -> None:
        """For an issue, find the adjacency mask and then execute the patch for
        each DataArray in the Dataset one by one.

        Args:
            issue (np.ndarray): 2d boolean array of the issue
        """
        adjacency_mask = self.find_adjacency(issue)
        patch_variables = [
            v
            for v in list(self.ds)
            if all(
                dim in self.ds[v].dims
                for dim in oet.config.config["analyze"]["lon_lat_dim"].split(",")
            )
        ]
        for variable in patch_variables:
            self.patch_one_variable(variable, issue, adjacency_mask)

    def patch_all_issues(self) -> xr.Dataset:
        """Find all issues in the Dataset. If there are any, then iterate over
        the issues and patch them one by one. If there are no issues, just
        return the dataset as it is.

        Returns:
            xr.Dataset: A copy of the dataset with patches included, or if there are no issues found, the
            original dataset.
        """
        issues = self.find_issues()
        oet.get_logger().warning(
            f"Finding {len(issues)} issues. N={[s.sum() for s in issues]}.",
        )
        if len(issues):
            self.ds = self.ds.copy()
            for issue in oet.utils.tqdm(
                issues,
                desc="patching issues",
                disable=len(issues) <= 1,
            ):
                self.execute_patch(issue)
            buffer = xr.zeros_like(self.ds['cell_area']).astype(bool)
            for issue_mask in issues:
                buffer.data = buffer.data | issue_mask
            self.ds['patched_data'] = buffer
            self.ds['patched_data'].attrs.update(
                description=f'{self.__class__.__name__} added data for {len(issues)} in these lat/lon places.',
            )
        self.ds.attrs.update({"Grid filler": f"Fixed {len(issues)}"})
        return self.ds
