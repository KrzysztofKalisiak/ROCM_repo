import numba as nb
import numpy as np
import torch

@nb.njit()
def haversine_distance(p1x,p1y,p2x,p2y):
    r = 6371.001
    p = np.radians(np.array([p1x, p1y, p2x, p2y]))
    return 2*r*np.arcsin(np.sqrt(
        (np.sin((p[1]-p[3])/2)**2)+(np.cos(p[1])*np.cos(p[3])*np.sin((p[0]-p[2])/2)**2)
        ))
@nb.njit()
def f(pct_n, shp_n):

    full_precompute = np.empty((pct_n.shape[0], shp_n.shape[0]), dtype=np.float32)

    for x in np.arange(pct_n.shape[0]):
        for y in np.arange(shp_n.shape[0]):
            full_precompute[x, y] = haversine_distance(
            pct_n[x, 0], pct_n[x, 1], shp_n[y, 0], shp_n[y, 1])
    return full_precompute

def HaversineLoss(output, y):

    loss = torch.mean(-torch.sum(torch.log(output) * y, 1))

    return loss