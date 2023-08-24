import os
import typing as ty
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import optim_esm_tools as oet
from optim_esm_tools.analyze.time_statistics import default_thresholds


class VariableMerger:
    full_paths = None
    source_files: ty.Mapping
    common_mask: xr.DataArray

    def __init__(self, paths, other_paths=None, merge_method='logical_or'):
        self.mask_paths = paths
        self.other_paths = other_paths or []
        self.merge_method = merge_method
        source_files, common_mask = self.process_masks()
        self.source_files = source_files
        self.common_mask = common_mask

    def squash_sources(self) -> xr.Dataset:
        common_mask = (
            self.common_mask
            if self.common_mask.dims == ('lat', 'lon')
            else oet.analyze.xarray_tools.reverse_name_mask_coords(self.common_mask)
        )

        new_ds = defaultdict(dict)
        new_ds['data_vars']['global_mask'] = common_mask
        for var, path in self.source_files.items():
            _ds = oet.load_glob(path)
            new_ds['data_vars'][var] = (
                _ds[var]
                .where(common_mask)
                .mean(oet.config.config['analyze']['lon_lat_dim'].split(','))
            )
            new_ds['data_vars'][var].attrs = _ds[var].attrs

        # Make one copy - just use the last dataset
        new_ds['data_vars']['cell_area'] = _ds['cell_area']
        keys = sorted(list(self.source_files.keys()))
        new_ds['attrs'] = dict(
            variables=keys,
            source_files=[self.source_files[k] for k in keys],
            mask_files=sorted(self.mask_paths),
        )
        try:
            new_ds = xr.Dataset(**new_ds)
        except TypeError as e:  # pragma: no cover
            oet.get_logger.warning(f'Ran into {e} fallback method because of cftime')
            # Stupid cftime can't compare it's own formats
            data_vars = new_ds.pop('data_vars')
            new_ds = xr.Dataset(**new_ds)

            # But xarray can fudge something along the way!
            for k, v in data_vars.items():
                new_ds[k] = v
        return new_ds

    def make_fig(self, new_ds=None, fig_kw=None):
        new_ds = new_ds or self.squash_sources()
        variables = list(new_ds.attrs['variables'])
        mapping = {str(i): v for i, v in enumerate(variables)}
        keys = list(mapping.keys()) + ['t']

        fig_kw = fig_kw or dict(
            mosaic=''.join(f'{k}.\n' for k in keys),
            figsize=(17, 4 * ((1 + len(keys)) / 3)),
            gridspec_kw=dict(width_ratios=[1, 1], wspace=0.1, hspace=0.05),
        )

        _, axes = plt.subplot_mosaic(**fig_kw)

        if len(keys) > 1:
            for k in keys[1:]:
                axes[k].sharex(axes[keys[0]])

        for key, var in mapping.items():
            plt.sca(axes[key])
            plot_kw = dict(label=var)
            oet.plotting.map_maker.plot_simple(new_ds, var, **plot_kw)
            plt.legend(loc='center left')

        ax = plt.gcf().add_subplot(
            1, 2, 2, projection=oet.plotting.plot.get_cartopy_projection()
        )
        oet.plotting.map_maker.overlay_area_mask(
            new_ds.where(new_ds['global_mask']).copy(), ax=ax
        )
        res_f, tips = result_table(new_ds)
        add_table(res_f=res_f, tips=tips, ax=axes['t'])

    def process_masks(self) -> ty.Tuple[dict, xr.DataArray]:
        source_files = {}
        common_mask = None
        for path in self.mask_paths:
            ds = oet.load_glob(path)
            # Source files may be non-unique!
            source_files[ds.attrs['variable_id']] = ds.attrs['file']
            common_mask = self.combine_masks(common_mask, ds)
        for other_path in self.other_paths:
            if other_path == '':  # pragma: no cover
                continue
            ds = oet.load_glob(other_path)
            # Source files may be non-unique!
            var = ds.attrs['variable_id']
            if var not in source_files:
                source_files[var] = ds.attrs['file']
        return source_files, common_mask

    def combine_masks(
        self,
        common_mask: ty.Optional[xr.DataArray],
        other_dataset: xr.Dataset,
        field: ty.Optional[str] = None,
    ) -> xr.DataArray:
        field = field or (
            'global_mask' if 'global_mask' in other_dataset else 'cell_area'
        )
        is_the_first_instance = common_mask is None
        if is_the_first_instance:
            return other_dataset[field].astype(np.bool_)
        if self.merge_method == 'logical_or':
            return common_mask | other_dataset[field].astype(np.bool_)

        raise NotImplementedError(f'No such method as {self.merge_method}')


def change_plt_table_height():
    """Increase the height of rows in plt.table

    Unfortunately, the options that you can pass to plt.table are insufficient to render a table
    that has rows with sufficient heights that work with a font that is not the default. From the
    plt.table implementation, I figured I could change these (rather patchy) lines in the source
    code:
    https://github.com/matplotlib/matplotlib/blob/b7dfdc5c97510733770429f38870a623426d0cdc/lib/matplotlib/table.py#L391

    Matplotlib version matplotlib==3.7.2
    """
    import matplotlib

    print('Change default plt.table row height')

    def _approx_text_height(self):
        return 1.5 * (
            self.FONTSIZE / 72.0 * self.figure.dpi / self._axes.bbox.height * 1.2
        )

    matplotlib.table.Table._approx_text_height = _approx_text_height


def add_table(res_f, tips, ax=None, fontsize=16, pass_color=(0.75, 1, 0.75)):
    ax = ax or plt.gcf().add_subplot(2, 2, 4)
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(
        cellText=res_f.values,
        rowLabels=res_f.index,
        colLabels=res_f.columns,
        cellColours=[
            [(pass_color if v else [1, 1, 1]) for v in row] for row in tips.values
        ],
        loc='bottom',
        colLoc='center',
        rowLoc='center',
        cellLoc='center',
    )
    table.set_fontsize(fontsize)


def result_table(ds, formats=None):
    res = {
        field: summarize_stats(ds, field, path)
        for field, path in zip(ds.attrs['variables'], ds.attrs['source_files'])
    }
    thrs = default_thresholds()
    is_tip = pd.DataFrame(
        {
            k: {
                t: (thrs[t][0](v, thrs[t][1]) if v is not None else False)
                for t, v in d.items()
            }
            for k, d in res.items()
        }
    ).T

    formats = formats or dict(
        n_breaks='.0f',
        p_symmetry='.3f',
        p_dip='.3f',
        max_jump='.1f',
        n_std_global='.1f',
    )
    res_f = pd.DataFrame(res).T
    for k, f in formats.items():
        res_f[k] = res_f[k].map(f'{{:,{f}}}'.format)

    order = list(formats.keys())
    return res_f[order], is_tip[order]


def summarize_stats(ds, field, path):
    return {
        'n_breaks': oet.analyze.time_statistics.calculate_n_breaks(ds, field=field),
        'p_symmetry': oet.analyze.time_statistics.calculate_symmetry_test(
            ds, field=field
        ),
        'p_dip': oet.analyze.time_statistics.calculate_dip_test(ds, field=field),
        'n_std_global': oet.analyze.time_statistics.n_times_global_std(
            ds=oet.load_glob(path).where(ds['global_mask'])
        ),
        'max_jump': oet.analyze.time_statistics.calculate_max_jump_in_std_history(
            ds=oet.load_glob(path).where(ds['global_mask']), mask=ds['global_mask']
        ),
    }


if __name__ == '__main__':
    change_plt_table_height()
