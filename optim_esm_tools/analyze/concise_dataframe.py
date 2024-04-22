import typing as ty

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import optim_esm_tools as oet


class ConciseDataFrame:
    delimiter: str = ', '
    merge_postfix: str = '(s)'

    def __init__(
        self,
        df: pd.DataFrame,
        group: ty.Optional[ty.Iterable] = None,
        tqdm: bool = False,
        match_overlap: bool = True,
        sort_by: ty.Union[str, ty.Tuple] = (
            'tips',
            'institution_id',
            'source_id',
            'experiment_id',
        ),
        match_by: ty.Iterable = ('institution_id', 'source_id', 'experiment_id'),
        min_frac_overlap: float = 0.33,
        eager_mode=True,
        disable_doubles: ty.Optional[ty.Iterable[str]] = None,
    ) -> None:
        # important to sort by tips == True first! As in match_rows there is a line that assumes
        # that all tipping rows are already merged!

        self.df = df.copy().sort_values(
            by=list(oet.utils.to_str_tuple(sort_by)),
            ascending=False,
        )
        self.group = group or (set(self.df.columns) - set(match_by))
        self.match_overlap = match_overlap
        self.tqdm = tqdm
        self.min_frac_overlap = min_frac_overlap
        self.eager_mode = eager_mode
        self.disable_doubles = disable_doubles

    def concise(self) -> pd.DataFrame:
        rows = [row.to_dict() for _, row in self.df.iterrows()]
        matched_rows = self.match_rows(rows)  # type: ignore
        combined_rows = [self.combine_rows(r, self.delimiter) for r in matched_rows]
        df_ret = pd.DataFrame(combined_rows)
        return self.rename_columns_with_plural(df_ret)

    def rename_columns_with_plural(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add postfix to columns from the dataframe."""
        rename_dict = {k: f'{k}{self.merge_postfix}' for k in self.group}
        return df.rename(columns=rename_dict)

    @staticmethod
    def combine_rows(rows: ty.Mapping, delimiter: str) -> ty.Dict[str, str]:
        ret = {}
        for k in rows[0].keys():
            try:
                vals = list({r[k] for r in rows})
                val = sorted(vals)
            except TypeError:
                val = sorted({str(v) for v in vals})
            ret[k] = val[0] if len(val) == 1 else delimiter.join([str(v) for v in val])
        return ret

    _mask_cache = None

    def overlaps_enough(self, path1, path2, use_field='global_mask'):
        self._mask_cache = self._mask_cache or {}
        for path in path1, path2:
            if path not in self._mask_cache:
                self._mask_cache[path] = oet.load_glob(path)[use_field].values
        result = (
            self.overlaps_percent(self._mask_cache[path1], self._mask_cache[path2])
            >= self.min_frac_overlap
        )
        if not self.eager_mode:
            self._mask_cache = None
        return result

    @staticmethod
    def overlaps_percent(arr1, arr2):
        return np.sum(arr1 & arr2) / min(np.sum(arr1), np.sum(arr2))

    def _row_is_double(self, row: pd.Series, other_row: pd.Series) -> bool:
        if not self.disable_doubles:
            return False

        return any(
            row.get(d, 'no d?') == other_row.get(d, 'also no?')
            for d in oet.utils.to_str_tuple(self.disable_doubles)
        )

    def _should_append_to_group(
        self,
        group: ty.List[pd.Series],
        other_row: pd.Series,
    ) -> bool:
        return (not self.match_overlap) or (
            any(
                self.overlaps_enough(r['path'], other_row['path'])
                for r in group
                if r['tips']
            )
        )

    def match_rows(self, rows):
        df = pd.DataFrame(rows)
        match = sorted(set(df.columns) - set(self.group))

        groups = []
        for row in oet.utils.tqdm(rows, desc='rows', disable=self.tqdm):
            if any(row in g for g in groups):
                continue

            groups.append([row])
            mask = np.ones(len(df), dtype=np.bool_)
            for m in match:
                mask &= df[m] == row[m]
            for other_row in oet.utils.tqdm(
                [row for m, row in zip(mask, rows) if m],
                desc='subrows',
                disable=self.tqdm,
            ):
                if row == other_row:
                    continue

                if self._row_is_double(row, other_row):
                    continue

                if self._should_append_to_group(groups[-1], other_row):
                    groups[-1].append(other_row)
        return groups
