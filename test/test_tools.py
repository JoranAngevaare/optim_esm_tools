import numpy as np
from hypothesis import example, given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.stats import percentileofscore

import optim_esm_tools as oet


# We used to have a "given" decorator here but it was too slow
def test_rank_2d_float():
    a = np.arange(2000).reshape(40, 50)
    _rank2d(a)
    a.astype(np.float64)
    _rank2d(a)


def _rank2d(a):
    nan_mask = np.isnan(a)
    a_flat = a[~nan_mask].flatten()

    if len(np.unique(a_flat)) < 2:
        return
    pcts = np.array(
        [[percentileofscore(a_flat, i, kind='mean') / 100 for i in aa] for aa in a],
    )
    rnk = oet.analyze.tools.rank2d(a)

    assert np.all(np.isclose(pcts, rnk, equal_nan=True))


@given(
    arrays(np.float16, shape=(50)).filter(
        lambda x: (np.isfinite(x).sum() > 40) and len(np.unique(x)) > 1,
    ),
)
def test_smooth_lowes_year(a):
    b = oet.analyze.tools.smoother_lowess_year(a)
    assert np.isfinite(a).sum() == len(b)
    assert np.isclose(np.nanmean(a[np.isfinite(a)]), np.nanmean(b), rtol=0.5, atol=0.5)
