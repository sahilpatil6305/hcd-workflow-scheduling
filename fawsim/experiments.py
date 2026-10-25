"""Experiment definitions and a reproducible, resumable runner."""
from __future__ import annotations

import hashlib
import json
import os
import zlib
from dataclasses import asdict, replace
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .environment import EnvConfig, V5_SPEEDS, make_environment, no_failure_environment
from .policies import make_policy
from .simulator import HorizonExceeded, simulate
from .workflows import load_wfcommons

MEAN_SPEED = float(np.mean(V5_SPEEDS))

INSTANCES = (
    "1000genome-chameleon-2ch-100k-001.json", "1000genome-chameleon-4ch-100k-001.json",
    "cycles-chameleon-1l-1c-9p-001.json", "cycles-chameleon-2l-1c-9p-001.json",
    "epigenomics-chameleon-hep-1seq-50k-001.json", "epigenomics-chameleon-ilmn-1seq-100k-001.json",
    "montage-chameleon-2mass-005d-001.json", "montage-chameleon-2mass-01d-001.json",
    "seismology-chameleon-100p-001.json", "seismology-chameleon-200p-001.json",
    "soykb-chameleon-10fastq-10ch-001.json", "soykb-chameleon-20fastq-10ch-001.json",
    "srasearch-chameleon-20a-001.json", "srasearch-chameleon-50a-001.json",
)
TUNING_SEEDS = tuple(range(0, 20))
TEST_SEEDS = tuple(range(1000, 1100))

_WF_CACHE: Dict[str, object] = {}
_M0_CACHE: Dict[tuple, float] = {}


def get_workflow(name: str):
    if name not in _WF_CACHE:
        _WF_CACHE[name] = load_wfcommons(name, MEAN_SPEED)
    return _WF_CACHE[name]


def failure_free_makespan(inst: str, speeds=V5_SPEEDS) -> float:
    key = (inst, tuple(speeds))
    if key not in _M0_CACHE:
        wf = get_workflow(inst)
        _M0_CACHE[key] = simulate(wf, no_failure_environment(speeds), make_policy("EFT"), prior_mean=0.0)["makespan"]
    return _M0_CACHE[key]


def stable_id(*parts) -> int:
    """Process-independent integer hash used to derive seed sequences."""
    return zlib.crc32(json.dumps(parts, sort_keys=True, default=str).encode()) & 0x7FFFFFFF


def run_replicate(inst: str, cfg: EnvConfig, seed: int, policies: Sequence[str],
                  env_key: str = "", prior_scale_for_all: float = 1.0) -> List[Dict]:
    """Run all policies on one (instance, environment, seed) with a shared trace."""
    wf = get_workflow(inst)
    m0 = failure_free_makespan(inst, cfg.speeds)
    key = asdict(cfg)
    if key.get("uptime_shape") == 1.0:      # field added in v5.1: keep v5.0 seeds unchanged
        key.pop("uptime_shape")
    seed_seq = [seed, stable_id(inst), stable_id(env_key or key)]
    horizon = 30.0 * m0 + 50.0
    rows = []
    for attempt in range(6):
        env = make_environment(cfg, seed_seq, horizon, m0=m0)
        try:
            rows = []
            for pname in policies:
                pol = make_policy(pname)
                res = simulate(wf, env, pol)
                res.pop("decision_log", None)
                rows.append({"instance": inst, "app": wf.app, "n_tasks": wf.n, "seed": seed,
                             "policy": pname, "m0": m0, "horizon": horizon,
                             "n_outage_intervals_run": sum(1 for v in env.trace.values() for a, b in v if a >= 0),
                             "run_volatile": " ".join(map(str, env.run_volatile)),
                             **{f"cfg_{k}": v for k, v in asdict(cfg).items() if k not in ("speeds", "uptime_shape")},
                             **({"cfg_uptime_shape": cfg.uptime_shape} if cfg.uptime_shape != 1.0 else {}),
                             **res})
            return rows
        except HorizonExceeded:
            horizon *= 4.0      # same seed sequence: the trace prefix is unchanged
    raise RuntimeError(f"horizon still exceeded for {inst} {cfg} seed {seed}")


def _worker(args):
    return run_replicate(*args)


def run_grid(jobs: Iterable[tuple], out_csv: str, processes: int = 2, chunksize: int = 4) -> None:
    """Run jobs (inst, cfg, seed, policies, env_key) in parallel; append rows to CSV."""
    import multiprocessing as mp
    import pandas as pd
    jobs = list(jobs)
    rows: List[Dict] = []
    with mp.Pool(processes) as pool:
        for k, r in enumerate(pool.imap(_worker, jobs, chunksize=chunksize)):
            rows.extend(r)
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(jobs)} replicates", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
