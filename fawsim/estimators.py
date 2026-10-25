"""Online failure-rate estimators fed only by monitoring events (time <= now).

Gamma-Poisson conjugate model for an exponential up-time process: with n observed
outage starts over U units of observed up-time, the posterior of lambda is
Gamma(a0 + n, b0 + U) and its mean (a0 + n)/(b0 + U) is used as a plug-in estimate.
Optional exponential forgetting decays n and U with time constant tau.
"""
from __future__ import annotations

import math
from typing import List, Optional


class RateEstimator:
    def __init__(self, m: int, prior_mean: float, a0: float = 1.0,
                 half_life: Optional[float] = None, pooled: bool = False):
        if prior_mean < 0 or a0 <= 0:
            raise ValueError("prior_mean >= 0 and a0 > 0 required")
        self.m = m
        self.a0 = a0
        self.prior_mean = prior_mean
        self.b0 = a0 / prior_mean if prior_mean > 0 else math.inf
        self.tau = (half_life / math.log(2.0)) if half_life else None
        self.pooled = pooled
        self.n = [0.0] * m
        self.U = [0.0] * m
        self.up = [True] * m
        self.t = [None] * m          # time of last update per VM (None = not started)

    # -- internal ---------------------------------------------------------------
    def _advance(self, j: int, t: float) -> None:
        if self.t[j] is None:
            self.t[j] = t
            return
        dt = t - self.t[j]
        if dt < -1e-9:
            raise ValueError("estimator received events out of order")
        dt = max(0.0, dt)
        if self.tau is None:
            if self.up[j]:
                self.U[j] += dt
        else:
            decay = math.exp(-dt / self.tau)
            self.n[j] *= decay
            self.U[j] = self.U[j] * decay + (self.tau * (1.0 - decay) if self.up[j] else 0.0)
        self.t[j] = t

    # -- monitoring interface -------------------------------------------------------
    def start(self, t: float, up_state: List[bool]) -> None:
        """Begin observation at time t with the given up/down state of each VM."""
        for j in range(self.m):
            self.t[j] = t
            self.up[j] = bool(up_state[j])

    def outage_start(self, j: int, t: float) -> None:
        self._advance(j, t)
        self.n[j] += 1.0
        self.up[j] = False

    def outage_end(self, j: int, t: float) -> None:
        self._advance(j, t)
        self.up[j] = True

    # -- queries ----------------------------------------------------------------
    def _counts(self, j: int, t: float):
        # exposure up to t without mutating state
        if self.t[j] is None:
            return self.n[j], self.U[j]
        dt = max(0.0, t - self.t[j])
        if self.tau is None:
            return self.n[j], self.U[j] + (dt if self.up[j] else 0.0)
        decay = math.exp(-dt / self.tau)
        return self.n[j] * decay, self.U[j] * decay + (self.tau * (1.0 - decay) if self.up[j] else 0.0)

    def rates(self, t: float) -> List[float]:
        if self.prior_mean == 0:
            return [0.0] * self.m
        if self.pooled:
            N = U = 0.0
            for j in range(self.m):
                n, u = self._counts(j, t)
                N += n
                U += u
            lam = (self.a0 + N) / (self.b0 + U)
            return [lam] * self.m
        out = []
        for j in range(self.m):
            n, u = self._counts(j, t)
            out.append((self.a0 + n) / (self.b0 + u))
        return out


class ChangeAwareEstimator:
    """History-aware estimator that drops stale history per VM (v5.1).

    Keeps two Gamma-Poisson estimators: `hist` (events before t=0) and `run` (events at
    t >= 0). For each VM it compares the two observed rates with a likelihood-ratio test for
    two Poisson rates,  LR = 2[n_h ln(n_h/(U_h l0)) + n_r ln(n_r/(U_r l0))],
    l0 = (n_h+n_r)/(U_h+U_r), against the chi-square(1) critical value `crit`.
    If LR > crit the history of that VM is judged stale and only (prior + run) data are
    used; otherwise history and run data are pooled.  O(1) per VM and query.
    """

    def __init__(self, m: int, prior_mean: float, a0: float = 1.0, crit: float = 6.635,
                 quasi: bool = False):
        self.m = m
        self.quasi = quasi                 # v5.1b: divide LR by dispersion phi (quasi-Poisson)
        self._last_end = [None] * m        # end of last history outage (start of current up period)
        self._ups: List[List[float]] = [[] for _ in range(m)]
        self.phi = 1.0
        self.last_up_since = [None] * m    # time each VM last came back up (any observed time)
        self.obs_start = None              # start of observation (censoring point for age)
        self.prior_mean = prior_mean
        self.a0 = a0
        self.b0 = a0 / prior_mean if prior_mean > 0 else math.inf
        self.crit = crit
        self.hist = RateEstimator(m, prior_mean, a0)
        self.run = RateEstimator(m, prior_mean, a0)
        self.run_started = False
        self.stale_flags = [False] * m   # last decision per VM (for diagnostics)

    def start(self, t: float, up_state: List[bool]) -> None:
        if self.obs_start is None:
            self.obs_start = t
        if t < 0:
            self.hist.start(t, up_state)
            # history window start is a censoring point: first up period is not recorded
        else:
            self._start_run(up_state)

    def _start_run(self, up_state=None) -> None:
        if self.run_started:
            return
        if up_state is None:
            up_state = list(self.hist.up) if self.hist.t[0] is not None else [True] * self.m
        self.run.start(0.0, up_state)
        self.run_started = True
        if self.quasi:
            self.phi = self.dispersion(self._ups)

    @staticmethod
    def dispersion(ups: List[List[float]], min_n: int = 3) -> float:
        """phi = max(1, mean over VMs of CV^2 of completed history up-times).
        For a renewal process the count variance is ~CV^2 x mean (Cox), so phi = 1 for Poisson."""
        cv2 = []
        for u in ups:
            if len(u) >= min_n:
                mu = sum(u) / len(u)
                var = sum((x - mu) ** 2 for x in u) / (len(u) - 1)
                if mu > 0:
                    cv2.append(var / (mu * mu))
        return max(1.0, sum(cv2) / len(cv2)) if cv2 else 1.0

    def outage_start(self, j: int, t: float) -> None:
        if t < 0:
            self.hist.outage_start(j, t)
            if self._last_end[j] is not None:
                self._ups[j].append(t - self._last_end[j])
        else:
            self._start_run()
            self.run.outage_start(j, t)

    def outage_end(self, j: int, t: float) -> None:
        self.last_up_since[j] = t
        if t < 0:
            self.hist.outage_end(j, t)
            self._last_end[j] = t
        else:
            self._start_run()
            self.run.outage_end(j, t)

    @staticmethod
    def lr_stat(nh: float, uh: float, nr: float, ur: float) -> float:
        if uh <= 0 or ur <= 0:
            return 0.0
        l0 = (nh + nr) / (uh + ur)
        if l0 <= 0:
            return 0.0
        s = 0.0
        if nh > 0:
            s += nh * math.log(nh / (uh * l0))
        if nr > 0:
            s += nr * math.log(nr / (ur * l0))
        return 2.0 * s

    def _counts(self, j: int, t: float):
        nh, uh = self.hist._counts(j, 0.0)
        if not self.run_started:
            return nh, uh
        nr, ur = self.run._counts(j, max(t, 0.0))
        if self.lr_stat(nh, uh, nr, ur) > self.crit * self.phi:
            self.stale_flags[j] = True
            return nr, ur
        self.stale_flags[j] = False
        return nh + nr, uh + ur

    def rates(self, t: float) -> List[float]:
        if self.prior_mean == 0:
            return [0.0] * self.m
        if t >= 0:
            self._start_run()
        out = []
        for j in range(self.m):
            n, u = self._counts(j, t)
            out.append((self.a0 + n) / (self.b0 + u))
        return out


def weibull_shape_from_cv2(cv2: float, lo: float = 0.2, hi: float = 5.0) -> float:
    """Solve Gamma(1+2/k)/Gamma(1+1/k)^2 - 1 = cv2 for the Weibull shape k (bisection)."""
    def f(k):
        return math.exp(math.lgamma(1 + 2 / k) - 2 * math.lgamma(1 + 1 / k)) - 1 - cv2
    if f(lo) < 0:
        return lo
    if f(hi) > 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pooled_cv2(ups: List[List[float]], min_n: int = 3) -> Optional[float]:
    vals = []
    for u in ups:
        if len(u) >= min_n:
            mu = sum(u) / len(u)
            var = sum((x - mu) ** 2 for x in u) / (len(u) - 1)
            if mu > 0:
                vals.append(var / (mu * mu))
    return sum(vals) / len(vals) if vals else None
