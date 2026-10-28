"""Earthquake location by a station network: synthetic data and the classical method.

Units: km, s, km/s. Axes: x, y horizontal (km), z depth (km, positive downwards).
Stations are at the surface (z = 0).

Velocity model: constant gradient, v(z) = v0 + g z. The travel time between two points then
has a closed form (rays are circular arcs):

    t = (1 / g) * arccosh( 1 + g^2 D^2 / (2 v(z1) v(z2)) ),     D = Euclidean distance

For S waves, v_S = v_P / sqrt(3) (Poisson solid), so the same formula holds with v0 and g
divided by sqrt(3).
"""
import math
from dataclasses import dataclass

import torch

VP0, GRAD = 5.5, 0.03                 # v_P(z) = 5.5 + 0.03 z  (km/s): 5.5 at the surface, 6.4 at 30 km
VPVS = math.sqrt(3.0)                 # v_P / v_S ratio
REGION = 100.0                        # square [0, 100] x [0, 100] km
ZMAX = 30.0                           # maximum earthquake depth (km)
SIGMA_P, SIGMA_S = 0.05, 0.10         # pick noise (s)
P_S_PICK = 0.7                        # probability that the S wave is picked


# ----------------------------------------------------------------------------- physics
def travel_time(src, sta, phase="P", v0=VP0, grad=GRAD):
    """Exact travel time in the gradient model. src (..., 3), sta (..., 3) in km."""
    if phase == "S":
        v0, grad = v0 / VPVS, grad / VPVS
    d2 = ((src - sta) ** 2).sum(-1)
    v1 = v0 + grad * src[..., 2]
    v2 = v0 + grad * sta[..., 2]
    if grad == 0:
        return d2.sqrt() / v0
    return torch.acosh(1.0 + grad**2 * d2 / (2.0 * v1 * v2)) / grad


def travel_time_numeric(src, sta, phase="P", v0=VP0, grad=GRAD, n=4000):
    """Same travel time, computed independently of the closed form: find the ray joining the
    two points by bisection, then integrate along it.

    Along a ray with parameter p = sin(i) / v (i: angle from the vertical, conserved by
    Snell's law) and v = v0 + g z, dz = cos(i) di / (p g), hence dx = sin(i) di / (p g) and
    dt = di / (g sin(i)): two integrals without singularity, even at the turning point
    (i = 90 degrees). Two families of rays: upgoing rays (nearby stations), and rays that
    dive below the source before coming back up (distant stations). src, sta: float64
    tensors (3,), src deeper than sta."""
    if phase == "S":
        v0, grad = v0 / VPVS, grad / VPVS
    if float(src[2]) < float(sta[2]):                  # travel time is symmetric: start from the deeper point
        src, sta = sta, src
    v1, v2 = v0 + grad * float(src[2]), v0 + grad * float(sta[2])
    x_target = float(torch.linalg.norm(src[:2] - sta[:2]))

    def ray(s):
        """s in ]0, 2[: s <= 1 upgoing ray (p = s / v1), s > 1 diving ray (p = (2 - s) / v1)."""
        p = (s if s <= 1 else 2 - s) / v1
        i1, i2 = math.asin(min(p * v1, 1.0)), math.asin(min(p * v2, 1.0))
        segs = [(i2, i1)] if s <= 1 else [(i2, math.pi / 2), (i1, math.pi / 2)]
        x = t = 0.0
        for a, b in segs:
            i = torch.linspace(a, b, n, dtype=torch.float64)
            x += torch.trapz(torch.sin(i), i).item() / (p * grad)
            t += torch.trapz(1.0 / torch.sin(i), i).item() / grad
        return x, t

    lo, hi = 1e-9, 2 - 1e-9
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ray(mid)[0] < x_target else (lo, mid)
    return ray(0.5 * (lo + hi))[1]


def ray_path(depth, distance, v0=VP0, grad=GRAD, n=100):
    """Ray path (in the source-station vertical plane) from a source at `depth` km to a
    surface station at `distance` km. In a constant gradient it is an arc of the circle
    centred at height z_c = -v0 / grad (above the surface). Returns (x, z), each (n,)."""
    if distance == 0:                                  # vertical ray
        z = torch.linspace(depth, 0.0, n, dtype=torch.float64)
        return torch.zeros_like(z), z
    zc = -v0 / grad
    xc = (distance**2 + zc**2 - (depth - zc) ** 2) / (2 * distance)
    radius = math.hypot(xc, depth - zc)
    t1, t2 = math.atan2(depth - zc, -xc), math.atan2(-zc, distance - xc)
    th = torch.linspace(t1, t2, n, dtype=torch.float64)
    return xc + radius * torch.cos(th), zc + radius * torch.sin(th)


# ----------------------------------------------------------------------------- networks
def make_network(n_stations=30, seed=0, region=REGION, min_dist=6.0):
    """Positions (n, 3) of surface stations, drawn at random with a minimum spacing."""
    g = torch.Generator().manual_seed(seed)
    pts = []
    while len(pts) < n_stations:
        p = torch.rand(2, generator=g, dtype=torch.float64) * region
        if all(torch.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    xy = torch.stack(pts)
    return torch.cat([xy, torch.zeros(n_stations, 1, dtype=torch.float64)], 1)


@dataclass
class Event:
    """An earthquake seen by a network: what we observe, and what we are looking for."""
    stations: torch.Tensor        # (n, 3) positions of the stations that recorded it
    station_ids: torch.Tensor     # (n,) indices in the full network
    t_p: torch.Tensor             # (n,) picked P arrival times (s)
    t_s: torch.Tensor             # (n,) picked S arrival times (s), 0 if not picked
    s_mask: torch.Tensor          # (n,) 1 if the S wave is picked
    hypo: torch.Tensor            # (3,) true hypocentre (km)
    t0: float                     # true origin time (s)

    def subset(self, keep):
        """The same earthquake, seen only by the stations `keep` (indices or mask)."""
        return Event(self.stations[keep], self.station_ids[keep], self.t_p[keep], self.t_s[keep],
                     self.s_mask[keep], self.hypo, self.t0)


def make_events(network, n_events, seed=1, drop=(0.0, 0.3), min_stations=6,
                depth=(1.0, ZMAX), region=REGION, sigma_p=SIGMA_P, sigma_s=SIGMA_S,
                p_s_pick=P_S_PICK, v0=VP0, grad=GRAD):
    """Random earthquakes (epicentre uniform in the region, uniform depth) recorded by the
    network, with a fraction of stations down drawn in `drop` for each earthquake, and noisy
    picks. The origin time is unknown to the observer.
    (v0, grad): the Earth's velocity model that generates the data."""
    g = torch.Generator().manual_seed(seed)
    n_sta = network.shape[0]
    events = []
    for _ in range(n_events):
        hypo = torch.cat([torch.rand(2, generator=g, dtype=torch.float64) * region,
                          depth[0] + torch.rand(1, generator=g, dtype=torch.float64)
                          * (depth[1] - depth[0])])
        t0 = float(torch.rand(1, generator=g) * 10.0)
        p_drop = drop[0] + (drop[1] - drop[0]) * float(torch.rand(1, generator=g))
        keep = torch.rand(n_sta, generator=g) >= p_drop
        if keep.sum() < min_stations:
            keep[torch.randperm(n_sta, generator=g)[:min_stations]] = True
        ids = keep.nonzero().squeeze(1)
        sta = network[ids]
        tp = t0 + travel_time(hypo, sta, "P", v0, grad) + sigma_p * torch.randn(len(ids), generator=g, dtype=torch.float64)
        ts = t0 + travel_time(hypo, sta, "S", v0, grad) + sigma_s * torch.randn(len(ids), generator=g, dtype=torch.float64)
        s_mask = (torch.rand(len(ids), generator=g) < p_s_pick).double()
        events.append(Event(sta, ids, tp, ts * s_mask, s_mask, hypo, t0))
    return events


# ----------------------------------------------------------------------------- classical location
def location_misfit(event, cand, sigma_p=SIGMA_P, sigma_s=SIGMA_S, v0=VP0, grad=GRAD):
    """Misfit of the classical location at candidate hypocentres cand (m, 3).

    For each candidate, the origin time is estimated by the median of the residuals
    (observed time - predicted travel time), and the misfit is the sum of
    |residual - t0| / sigma (L1 norm, robust to outlier picks).
    Returns (misfit (m,), origin time (m,))."""
    sta = event.stations
    sta_s = sta[event.s_mask > 0]
    obs = torch.cat([event.t_p, event.t_s[event.s_mask > 0]])
    w = torch.cat([torch.full_like(event.t_p, 1 / sigma_p),
                   torch.full((sta_s.shape[0],), 1 / sigma_s, dtype=torch.float64)])
    tt = torch.cat([travel_time(cand[:, None, :], sta[None], "P", v0, grad),
                    travel_time(cand[:, None, :], sta_s[None], "S", v0, grad)], 1)
    res = obs[None] - tt
    t0 = res.median(dim=1, keepdim=True).values
    return ((res - t0).abs() * w).sum(1), t0[:, 0]


def grid_search_locate(event, margin=20.0, zmax=40.0, steps=(4.0, 1.0, 0.25), sigma_p=SIGMA_P,
                       sigma_s=SIGMA_S, v0=VP0, grad=GRAD):
    """Classical location by grid search, with the velocity model (v0, grad).

    Coarse grid (step steps[0]) over the station area widened by `margin`, from 0 to zmax in
    depth, then finer and finer grids around the best point.
    Returns (hypocentre (3,), origin time)."""
    sta = event.stations

    def box(lo, hi, h):
        axes = [torch.arange(a, b + 1e-9, h, dtype=torch.float64) for a, b in zip(lo, hi)]
        g = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([a.reshape(-1) for a in g], 1)

    lo = torch.cat([sta[:, :2].min(0).values - margin, torch.zeros(1, dtype=torch.float64)])
    hi = torch.cat([sta[:, :2].max(0).values + margin, torch.tensor([zmax], dtype=torch.float64)])
    best = None
    for i, h in enumerate(steps):
        if i == 0:
            cand = box(lo.tolist(), hi.tolist(), h)
        else:
            half = steps[i - 1]
            cand = box(torch.maximum(best - half, lo).tolist(), torch.minimum(best + half, hi).tolist(), h)
        m, t0 = location_misfit(event, cand, sigma_p, sigma_s, v0, grad)
        j = m.argmin()
        best, best_t0 = cand[j], t0[j].item()
    return best, best_t0


def azimuthal_gap(event):
    """Largest azimuth gap (degrees) between two consecutive stations seen from the epicentre."""
    d = event.stations[:, :2] - event.hypo[:2]
    az = torch.sort(torch.rad2deg(torch.atan2(d[:, 1], d[:, 0])) % 360).values
    gaps = torch.diff(torch.cat([az, az[:1] + 360]))
    return float(gaps.max())
