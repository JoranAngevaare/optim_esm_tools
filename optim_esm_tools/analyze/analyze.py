"""Scaffolding"""

import optim_esm_tools as oet

import pandas as pd
import os
from collections import defaultdict
from functools import wraps
from optim_esm_tools.synda_files import format_synda as st
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def requires(*requirements, do_raise=True):
    def somedec_outer(fn):
        @wraps(fn)
        def somedec_inner(self, *args, **kwargs):
            missing = []
            for r in requirements:
                if getattr(self, r, None) is None:
                    missing.append(r)
            message = f'Missing \n\t' + "\n\t".join(missing)
            if do_raise and missing:
                raise ValueError(message)
            elif missing:
                print(message)
            response = fn(self, *args, **kwargs)
            return response

        return somedec_inner

    return somedec_outer


class TippingCondition:

    @requires('pi_control', 'historical', 'scenario', do_raise=False)
    def i_mean_difference(self) -> dict:
        data_sets = 'pi_control', 'historical', 'scenario'
        result = {}
        for ds in data_sets:
            detrend = self.detrend(self.getattr(ds))
            result[ds] = (detrend[-1] - detrend[0]) / detrend[0]
        return result

    @requires('pi_control', 'historical', 'scenario', do_raise=False)
    def ii_std_difference(self) -> dict:
        data_sets = 'pi_control', 'historical', 'scenario'
        result = {}
        detrend_historical = self.detrend(self.historical)
        for ds in data_sets:
            detrend = self.detrend(self.getattr(ds))
            result[ds] = np.max(np.abs(detrend - np.mean(detrend)) / np.std(detrend_historical))
        return result

    @requires('pi_control', 'historical', 'scenario', do_raise=False)
    def iii_10yr_difference(self) -> dict:
        data_sets = 'pi_control', 'historical', 'scenario'
        result = {}
        for ds in data_sets:
            detrend = self.detrend(self.getattr(ds))
            result[ds] = np.max(detrend[10:] - detrend[:-10]) / np.mean(detrend)
        return result

    @property
    def i(self):
        return self.i_mean_difference()

    @property
    def ii(self):
        return self.ii_std_difference()

    @property
    def iii(self):
        return self.iii_10yr_difference()

    def time_series(self, dataset):
        ds = getattr(self, dataset)
        data = ds[self.parameter].values
        result = {
            'time': ds['year'].values,
            self.parameter: data,
            f'detrend_{self.parameter}': self.detrend(ds),
            'detrend_time': self._ma(ds['year'].values),
        }

        return result

    def show_time_series(self, dataset):
        data = self.time_series(dataset)
        plt.plot(data['time'], data[self.parameter], label='yearly')
        filter_width = self.filter_width
        plt.plot(data['detrend_time'], data[f'detrend_{self.parameter}'], label='detrend')
        plt.legend()
        plt.ylabel(self.parameter)

    def getattr(self, k):
        return getattr(self, k)


class TippingBase:
    def __init__(self, pos, parameter, pi_control=None, historical=None, scenario=None,
                 filter_width=10, ):
        self.pos = pos
        self.parameter = parameter
        self.filter_width = filter_width
        self.pi_control = self.xr_open(pi_control)
        self.historical = self.xr_open(historical)
        self.scenario = self.xr_open(scenario)
        self.check(do_raise=False)

    def check(self, do_raise=True):
        for attr in 'pi_control historical scenario'.split():
            if getattr(self, attr, None) is None:
                print(f'{attr} is None')
                if do_raise:
                    raise

    def detrend(self, ds):
        array = ds[self.parameter].values
        return self._ma(array)

    def _ma(self, a):
        return moving_average(a, n=self.filter_width)

    def xr_open(self, base):
        return xr.open_mfdataset(os.path.join(base, self.pos)).load()


class Tipping(TippingBase, TippingCondition):
    pass

# if __name__ == '__main__':
#
#     results = []
#     for pos in tqdm.tqdm(positions):
#         tip = Tipping(pos=pos, parameter='tas', **kw)
#         res = dict(pos=pos)
#         for k in 'i ii iii'.split():
#             sub_res = getattr(tip, k)
#             for kk, vv in sub_res.items():
#                 res[f'{k}_{kk}'] = vv
#         results.append(res)
#
#
#     tip = Tipping(pos=pos, parameter='tas', **kw, )
#     _, axes = plt.subplots(3, 1, gridspec_kw=dict(hspace=0.3), figsize=(8, 8))
#     for ax, ds in zip(axes, 'pi_control historical scenario'.split()):
#         plt.sca(ax)
#         plt.text(0.03, 0.95,
#                  ds,
#                  color='black',
#                  va='top',
#                  ha='left',
#                  transform=ax.transAxes,
#                  bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round'))
#
#         tip.show_time_series(ds)
#         plt.legend(loc='upper right', ncol=2)
#
#     kw = dict(range=[-0.01, 0.05], bins=100)
#     plt_kw = dict(drawstyle='steps-mid')
#     for label in ['i_pi_control', 'i_historical', 'i_scenario']:
#         c, be = np.histogram(df[label], **kw)
#         be *= 100
#         plt.plot((be[1:] + be[:-1]) / 2, c / np.max(c), label=label, **plt_kw)
#     plt.title('Difference start and end')
#     plt.xlabel('Difference [$\%$]')
#     plt.ylabel('Scaled probability')
#     plt.legend()
#
#     kw = dict(range=[0, 10],
#               bins=100)
#     plt_kw = dict(drawstyle='steps-mid')
#     for label in ['ii_pi_control', 'ii_historical', 'ii_scenario']:
#         c, be = np.histogram(df[label], **kw)
#         plt.plot((be[1:] + be[:-1]) / 2, c / np.max(c), label=label, **plt_kw)
#     plt.title('Max $\sigma$ from mean')
#     plt.xlabel('$\sigma$')
#     plt.ylabel('Scaled probability')
#     plt.legend()
