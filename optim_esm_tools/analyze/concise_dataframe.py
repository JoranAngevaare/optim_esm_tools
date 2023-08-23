from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import optim_esm_tools as oet


class ConciseDataFrame:
    delimiter = ', '
    min_frac_overlap = 0.33
    merge_postfix = '(s)'

    def __init__(self, df, group=None, tqdm=False):
        # important to sort by tips == True first! As in match_rows there is a line that assumes
        # that all tipping rows are already merged!
        self.df = df.copy().sort_values(
            by=['tips', 'institution_id', 'source_id', 'experiment_id'], ascending=False
        )
        self.group = group or (
            'method',
            'cluster',
            'variant_label',
            'cluster_i',
            'variable_id',
            'version',
            'figure',
        )
        self.match_overlap = True
        self.tqdm = tqdm

    def concise(self):
        rows = [row.to_dict() for _, row in self.df.iterrows()]
        matched_rows = self.match_rows(rows)
        combined_rows = [self.combine_rows(r, self.delimiter) for r in matched_rows]
        df_ret = pd.DataFrame(combined_rows)
        return self.rename_s(df_ret)

    def rename_s(self, df):
        rename_dict = {k: f'{k}{self.merge_postfix}' for k in self.group}
        return df.rename(columns=rename_dict)

    @staticmethod
    def combine_rows(rows, delimiter):
        ret = {}
        for k in rows[0].keys():
            val = sorted(list(set(r[k] for r in rows)))
            if len(val) == 1:
                ret[k] = val[0]
            else:
                ret[k] = delimiter.join([str(v) for v in val])
        return ret

    def match_rows(self, rows):
        groups = []
        for row in oet.utils.tqdm(rows, desc='rows', disable=not self.tqdm):
            if any(row in g for g in groups):
                continue

            groups.append([row])
            for other_row in rows:
                if row == other_row:
                    continue
                for k, v in row.items():
                    if k in self.group:
                        continue
                    if other_row.get(k) != v:
                        break
                else:
                    if (not self.match_overlap) or (
                        any(
                            self.overlaps_enough(r['path'], other_row['path'])
                            for r in groups[-1]
                            if r['tips']
                        )
                    ):
                        groups[-1].append(other_row)

        return groups

    @staticmethod
    def overlaps_percent(ds1, ds2, use_field='global_mask'):
        arr1 = ds1[use_field].values
        arr2 = ds2[use_field].values
        return np.sum(arr1 & arr2) / min(np.sum(arr1), np.sum(arr2))

    def overlaps_enough(self, path1, path2):
        return (
            self.overlaps_percent(oet.load_glob(path1), oet.load_glob(path2))
            >= self.min_frac_overlap
        )
