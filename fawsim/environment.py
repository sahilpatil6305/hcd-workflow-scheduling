"""Failure environments: per-VM rate profiles and exogenous outage traces.

An outage trace is a dict vm -> sorted list of disjoint (start, end) intervals covering
[-H, horizon). Intervals before 0 form the monitoring history; the simulator only
reveals events with time <= now to schedulers that are allowed to see them.
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

V5_SPEEDS: Tuple[float, ...] = (1.00, 1.20, 0.82, 1.45, 0.95, 1.12, 0.72, 1.30)
COUPLINGS = ("uniform", "fast-volatile", "random-volatile", "slow-volatile")

Interval = Tuple[float, float]
Trace = Dict[int, List[Interval]]


@dataclass(frozen=True)
class EnvConfig:
    rho: float                      # pool-mean rate x mean task duration (d_bar = 1)
    coupling: str = "uniform"
    speeds: Tuple[float, ...] = V5_SPEEDS
    n_volatile: int = 2
    volatile_mult: float = 10.0     # volatile rate / stable rate
    repair: float = 1.0             # fixed down time R
    overhead: float = 0.05          # restart overhead o
    history_k: float = 0.0          # history window H = history_k / pool-mean rate
    regime_shift: bool = False      # history volatile set disjoint from run volatile set
    shock: Optional[Tuple[int, float, float]] = None  # (n_vms_hit, start_frac_of_M0, dur in time units)
    uptime_shape: float = 1.0       # Weibull shape of up-times (1 = exponential); mean kept at 1/rate

    @property
    def m(self) -> int:
        return len(self.speeds)


def volatile_set(cfg: EnvConfig, rng: np.random.Generator) -> Tuple[int, ...]:
    m, k = cfg.m, cfg.n_volatile
    order = sorted(range(m), key=lambda j: (cfg.speeds[j], j))
    if cfg.coupling == "uniform":
        return ()
    if cfg.coupling == "fast-volatile":
        return tuple(sorted(order[-k:]))
    if cfg.coupling == "slow-volatile":
        return tuple(sorted(order[:k]))
    if cfg.coupling == "random-volatile":
        return tuple(sorted(int(x) for x in rng.choice(m, size=k, replace=False)))
    raise ValueError(f"unknown coupling {cfg.coupling}")


def rates_for(cfg: EnvConfig, vol: Sequence[int]) -> List[float]:
    """Per-VM rates with pool mean exactly rho (d_bar = 1)."""
    m = cfg.m
    lam_bar = cfg.rho
    if not vol:
        return [lam_bar] * m
    k = len(vol)
    lam_s = lam_bar * m / ((m - k) + k * cfg.volatile_mult)
    return [lam_s * cfg.volatile_mult if j in vol else lam_s for j in range(m)]


@dataclass
class Environment:
    cfg: EnvConfig
    run_rates: List[float]
    hist_rates: List[float]
    run_volatile: Tuple[int, ...]
    hist_volatile: Tuple[int, ...]
    history_start: float            # -H (0 if no history)
    trace: Trace = field(repr=False)
    horizon: float = 0.0
    shock_info: Optional[Tuple[Tuple[int, ...], float, float]] = None

    def true_rates(self, t: float) -> List[float]:
        """Oracle information: the hazard regime in force at time t (not outage times)."""
        return list(self.hist_rates if t < 0 else self.run_rates)


def _draw_up(rng: np.random.Generator, rate: float, shape: float) -> float:
    if shape == 1.0:
        return float(rng.exponential(1.0 / rate))
    scale = (1.0 / rate) / math.gamma(1.0 + 1.0 / shape)
    return float(scale * rng.weibull(shape))


def _renewal(rng: np.random.Generator, rate_hist: float, rate_run: float,
             t0: float, horizon: float, repair: float, shape: float = 1.0) -> List[Interval]:
    """Alternating renewal: Exp up-times (piecewise-constant rate switching at t=0), fixed repair.

    Exact sampling for a piecewise-constant hazard uses memorylessness: if an up-time
    drawn under the history rate crosses t=0, the remainder is redrawn at the run rate.
    """
    out: List[Interval] = []
    t = t0
    while True:
        rate = rate_hist if t < 0 else rate_run
        if rate <= 0:
            if t < 0 and rate_run > 0:
                t = 0.0
                continue
            break
        u = _draw_up(rng, rate, shape)
        if t < 0 and t + u >= 0:
            # crossed the regime boundary while up: restart the clock at 0
            # (exact for exponential; an approximation for Weibull up-times).
            t = 0.0
            if rate_run <= 0:
                break
            u = _draw_up(rng, rate_run, shape)
        start = t + u
        if start >= horizon:
            break
        out.append((start, start + repair))
        t = start + repair
    return out


def merge(intervals: Sequence[Interval]) -> List[Interval]:
    iv = sorted((float(a), float(b)) for a, b in intervals if b > a)
    out: List[List[float]] = []
    for a, b in iv:
        if out and a <= out[-1][1] + 1e-12:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def make_environment(cfg: EnvConfig, seed_seq: Sequence[int], horizon: float,
                     m0: float = 1.0) -> Environment:
    """Deterministic environment for a replicate. `m0` scales the shock start time."""
    ss = np.random.SeedSequence(list(seed_seq))
    child = ss.spawn(cfg.m + 2)
    rng_cfg = np.random.default_rng(child[0])
    run_vol = volatile_set(cfg, rng_cfg)
    if cfg.regime_shift:
        others = [j for j in range(cfg.m) if j not in run_vol]
        hist_vol = tuple(sorted(int(x) for x in rng_cfg.choice(others, size=cfg.n_volatile, replace=False)))
    else:
        hist_vol = run_vol
    run_rates = rates_for(cfg, run_vol)
    hist_rates = rates_for(cfg, hist_vol)
    H = cfg.history_k / cfg.rho if (cfg.history_k > 0 and cfg.rho > 0) else 0.0
    trace: Trace = {}
    for j in range(cfg.m):
        rng = np.random.default_rng(child[2 + j])
        trace[j] = _renewal(rng, hist_rates[j], run_rates[j], -H, horizon, cfg.repair, cfg.uptime_shape)
    shock_info = None
    if cfg.shock is not None:
        n_hit, frac, dur = cfg.shock
        rng_s = np.random.default_rng(child[1])
        hit = tuple(sorted(int(x) for x in rng_s.choice(cfg.m, size=n_hit, replace=False)))
        s0 = frac * m0
        for j in hit:
            trace[j] = merge(trace[j] + [(s0, s0 + dur)])
        shock_info = (hit, s0, dur)
    trace = {j: merge(v) for j, v in trace.items()}
    return Environment(cfg, run_rates, hist_rates, run_vol, hist_vol, -H, trace, horizon, shock_info)


def no_failure_environment(speeds: Sequence[float]) -> Environment:
    cfg = EnvConfig(rho=0.0, speeds=tuple(speeds))
    m = len(speeds)
    return Environment(cfg, [0.0] * m, [0.0] * m, (), (), 0.0, {j: [] for j in range(m)}, float("inf"))
