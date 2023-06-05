import numba
import numpy as np


def _mask2d_to_xy_slice(mask: np.array, cyclic:bool=False)->np.array:
    where = np.argwhere(mask)
    slices = np.zeros((len(mask), 2, 2), dtype=np.int64)
    n_slices = 1
    slices[0][0][0] = where[0][0]
    slices[0][0][1] = where[0][0] + 1
    slices[0][1][0] = where[0][1]
    slices[0][1][1] = where[0][1] + 1

    for x, y in where[1:]:
        # x1 and y1 are EXLCUSIVE!
        for s_i, ((x0, x1), (y0, y1)) in enumerate(slices[:n_slices]):
            if x0 <= x <= x1 and y0 <= y <= y1:
                if x == x1:
                    slices[s_i][0][1] += 1
                if y == y1:
                    slices[s_i][1][1] += 1
                if cyclic:
                    raise ValueError
                break
        else:
            slices[n_slices][0][0] = x
            slices[n_slices][0][1] = x + 1

            slices[n_slices][1][0] = y
            slices[n_slices][1][1] = y + 1

            n_slices += 1
    return slices[:n_slices]


mask2d_to_xy_slice = numba.njit(_mask2d_to_xy_slice)
