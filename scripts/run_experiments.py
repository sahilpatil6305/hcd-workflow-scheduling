"""Experiment entry point.  Usage:
    python scripts/run_experiments.py tuning     # seeds 0-19: pipeline check and RT beta grid
    python scripts/run_experiments.py choose_beta
    python scripts/run_experiments.py main       # E1: seeds 1000-1099
    python scripts/run_experiments.py history    # E2: history depth (stationary + regime shift)
    python scripts/run_experiments.py sensitivity
    python scripts/run_experiments.py shock
All raw rows are written to results/<name>/raw.csv together with config.json.
"""
import json
import os
import platform
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fawsim import __version__
from fawsim.environment import COUPLINGS, EnvConfig
from fawsim.experiments import INSTANCES, TEST_SEEDS, TUNING_SEEDS, file_sha256, run_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHOS = (0.01, 0.03, 0.1, 0.3)
BETAS = (0.5, 1, 2, 4, 8)
E1_POLICIES = ["EFT", "ECT-Oracle", "ECT-OraclePooled", "ECT-Online", "ECT-OnlinePooled", "AR-v4"]
# Frozen after tuning (see docs/RESEARCH_DESIGN.md section 10, deviation D1):
B = 0.5
E1_FINAL = E1_POLICIES + [f"RT-Oracle-b{B:g}", f"RT-OraclePooled-b{B:g}", f"RT-Online-b{B:g}",
                          f"RT-OnlinePooled-b{B:g}", "ECTR-Oracle", "ECTR-Online"]


def save_config(name, payload):
    d = os.path.join(ROOT, "results", name)
    os.makedirs(d, exist_ok=True)
    payload = dict(payload)
    payload.update({"fawsim_version": __version__, "python": platform.python_version(),
                    "numpy": np.__version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "instances_sha256": {i: file_sha256(os.path.join(ROOT, "data", "wfcommons", i)) for i in INSTANCES}})
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    return os.path.join(d, "raw.csv")


def tuning():
    pols = E1_POLICIES + [f"RT-Oracle-b{b:g}" for b in BETAS]
    jobs = [(i, EnvConfig(rho=r, coupling=c), s, pols, "") for r in RHOS for c in COUPLINGS
            for i in INSTANCES for s in TUNING_SEEDS]
    out = save_config("tuning", {"seeds": TUNING_SEEDS, "rhos": RHOS, "couplings": COUPLINGS,
                                 "policies": pols, "n_jobs": len(jobs)})
    run_grid(jobs, out)


def tuning_modes():
    """Second tuning pass (added after the first pass showed the pre-registered ECT rule
    under-performing EFT): compare ECT modes on tuning seeds only."""
    pols = ["EFT", "ECTF-Oracle", "ECTB-Oracle", "ECTR-Oracle", "ECTF-Online", "ECTB-Online",
            "ECTR-Online", "RT-Oracle-b0.5", "RT-Oracle-b1"]
    jobs = [(i, EnvConfig(rho=r, coupling=c), s, pols, "") for r in RHOS for c in COUPLINGS
            for i in INSTANCES for s in TUNING_SEEDS]
    out = save_config("tuning_modes", {"seeds": TUNING_SEEDS, "rhos": RHOS, "couplings": COUPLINGS,
                                       "policies": pols, "n_jobs": len(jobs)})
    run_grid(jobs, out)


def tuning_rt_online():
    """Third tuning pass: beta for RT with online / pooled rate estimates (tuning seeds only)."""
    pols = ["EFT"] + [f"RT-{src}-b{b:g}" for src in ("Online", "OraclePooled") for b in (0.25, 0.5, 1, 2)] \
        + ["RT-Oracle-b0.25"]
    jobs = [(i, EnvConfig(rho=r, coupling=c), s, pols, "") for r in RHOS for c in COUPLINGS
            for i in INSTANCES for s in TUNING_SEEDS]
    out = save_config("tuning_rt_online", {"seeds": TUNING_SEEDS, "policies": pols, "n_jobs": len(jobs)})
    run_grid(jobs, out)


def choose_beta():
    """Select RT beta on tuning seeds: min mean log(makespan/EFT) over all tuning replicates."""
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "results", "tuning", "raw.csv"))
    w = df.pivot_table(index=["instance", "seed", "cfg_rho", "cfg_coupling"], columns="policy", values="makespan")
    lr = np.log(w.div(w["EFT"], axis=0))
    rt = lr[[c for c in lr.columns if c.startswith("RT-Oracle-b")]].mean().sort_values()
    beta = float(rt.index[0].split("-b")[1])
    with open(os.path.join(ROOT, "results", "tuning", "chosen_beta.json"), "w") as fh:
        json.dump({"beta": beta, "criterion": "min mean log(makespan/EFT) over all tuning replicates "
                   "(seeds 0-19, 4 rho x 4 couplings x 14 instances)", "scores": rt.to_dict()}, fh, indent=1)
    print("chosen beta", beta, rt.round(4).to_dict())


E5_GRID = [(c, r, k, sh) for c in ("random-volatile", "fast-volatile") for r in (0.03, 0.1, 0.3)
           for k in (5, 50) for sh in (False, True)]


def tuning_cd():
    """v5.1: choose alpha for stale-history detection on tuning seeds only."""
    pols = ["EFT", f"RT-Hist-b{B:g}"] + [f"RT-HistCD-b{B:g}-a{a:g}" for a in (0.001, 0.01, 0.05)]
    jobs = [(i, EnvConfig(rho=r, coupling=c, history_k=k, regime_shift=sh), s, pols, "")
            for c, r, k, sh in E5_GRID for i in INSTANCES for s in TUNING_SEEDS]
    out = save_config("tuning_cd", {"seeds": TUNING_SEEDS, "policies": pols, "grid": E5_GRID, "n_jobs": len(jobs)})
    run_grid(jobs, out)
    import pandas as pd
    df = pd.read_csv(out)
    w = df.pivot_table(index=["instance", "seed", "cfg_rho", "cfg_coupling", "cfg_history_k", "cfg_regime_shift"],
                       columns="policy", values="makespan")
    lr = np.log(w.div(w["EFT"], axis=0))
    cd = lr[[c for c in lr.columns if "HistCD" in c]].mean().sort_values()
    alpha = float(cd.index[0].rsplit("-a", 1)[1])
    with open(os.path.join(ROOT, "results", "tuning_cd", "chosen_alpha.json"), "w") as fh:
        json.dump({"alpha": alpha, "scores": cd.to_dict(),
                   "by_shift": lr.groupby(level="cfg_regime_shift").mean().to_dict()}, fh, indent=1, default=str)
    print("chosen alpha", alpha, cd.round(5).to_dict())


def e5():
    with open(os.path.join(ROOT, "results", "tuning_cd", "chosen_alpha.json")) as fh:
        a = json.load(fh)["alpha"]
    pols = ["EFT", f"RT-Oracle-b{B:g}", f"RT-Online-b{B:g}", f"RT-Hist-b{B:g}", f"RT-HistDisc-b{B:g}",
            f"RT-HistCD-b{B:g}-a{a:g}", "ECT-Hist", f"ECT-HistCD-a{a:g}"]
    jobs = [(i, EnvConfig(rho=r, coupling=c, history_k=k, regime_shift=sh), s, pols, "")
            for c, r, k, sh in E5_GRID for i in INSTANCES for s in TEST_SEEDS]
    out = save_config("e5_stale_history", {"seeds": TEST_SEEDS, "policies": pols, "alpha": a,
                                           "grid": E5_GRID, "n_jobs": len(jobs)})
    run_grid(jobs, out)


E6_GRID = [(sh_, c, r, k, sh) for sh_ in (1.0, 0.7, 0.5) for c in ("random-volatile", "fast-volatile")
           for r in (0.03, 0.1) for k in (5, 50) for sh in (False, True)]


def e6():
    with open(os.path.join(ROOT, "results", "tuning_cd", "chosen_alpha.json")) as fh:
        a = json.load(fh)["alpha"]
    pols = ["EFT", f"RT-Oracle-b{B:g}", f"RT-Online-b{B:g}", f"RT-Hist-b{B:g}", f"RT-HistDisc-b{B:g}",
            f"RT-HistCD-b{B:g}-a{a:g}", f"RT-HistCDQ-b{B:g}-a{a:g}"]
    jobs = [(i, EnvConfig(rho=r, coupling=c, history_k=k, regime_shift=sh, uptime_shape=shp), s, pols, "")
            for shp, c, r, k, sh in E6_GRID for i in INSTANCES for s in TEST_SEEDS]
    out = save_config("e6_bursty", {"seeds": TEST_SEEDS, "policies": pols, "alpha": a,
                                    "grid": E6_GRID, "n_jobs": len(jobs)})
    run_grid(jobs, out)


def tuning_age():
    """v5.2 pilot on tuning seeds: age-aware Weibull RT vs RT-HistCD (no new tuned parameter)."""
    pols = ["EFT", f"RT-Oracle-b{B:g}", f"RT-Online-b{B:g}", f"RT-HistCD-b{B:g}-a0.05", f"RT-HistCDA-b{B:g}-a0.05"]
    jobs = [(i, EnvConfig(rho=r, coupling=c, history_k=k, regime_shift=sh, uptime_shape=shp), s, pols, "")
            for shp, c, r, k, sh in E6_GRID for i in INSTANCES for s in TUNING_SEEDS]
    out = save_config("tuning_age", {"seeds": TUNING_SEEDS, "policies": pols, "grid": E6_GRID, "n_jobs": len(jobs)})
    run_grid(jobs, out)


def chosen_beta():
    with open(os.path.join(ROOT, "results", "tuning", "chosen_beta.json")) as fh:
        return json.load(fh)["beta"]


def main():
    b = chosen_beta()
    assert b == B, "tuned beta changed; update E1_FINAL deliberately"
    pols = E1_FINAL
    jobs = [(i, EnvConfig(rho=r, coupling=c), s, pols, "") for r in RHOS for c in COUPLINGS
            for i in INSTANCES for s in TEST_SEEDS]
    out = save_config("main", {"seeds": TEST_SEEDS, "rhos": RHOS, "couplings": COUPLINGS,
                               "policies": pols, "rt_beta_from_tuning": b, "n_jobs": len(jobs)})
    run_grid(jobs, out)


def history():
    pols = ["EFT", "ECT-Oracle", "ECT-Online", "ECT-Hist", "ECT-HistDisc",
            f"RT-Oracle-b{B:g}", f"RT-Online-b{B:g}", f"RT-Hist-b{B:g}", f"RT-HistDisc-b{B:g}"]
    jobs = []
    for k in (5, 50):
        for shift in (False, True):
            for r in (0.1, 0.3):
                cfg = EnvConfig(rho=r, coupling="random-volatile", history_k=k, regime_shift=shift)
                jobs += [(i, cfg, s, pols, "") for i in INSTANCES for s in TEST_SEEDS]
    out = save_config("history", {"seeds": TEST_SEEDS, "policies": pols, "history_k": (5, 50),
                                  "regime_shift": (False, True), "rhos": (0.1, 0.3),
                                  "coupling": "random-volatile", "n_jobs": len(jobs)})
    run_grid(jobs, out)


def sensitivity():
    jobs = []
    base = dict(rho=0.1, coupling="fast-volatile")
    pols = ["EFT", "ECT-Oracle", "ECT-Online", "ECT-Online-prior0.25", "ECT-Online-prior4",
            f"RT-Oracle-b{B:g}", f"RT-Online-b{B:g}", f"RT-Online-b{B:g}-prior0.25", f"RT-Online-b{B:g}-prior4"]
    variants = {
        "repair0.25": EnvConfig(**base, repair=0.25),
        "repair4": EnvConfig(**base, repair=4.0),
        "volmult3": EnvConfig(**base, volatile_mult=3.0),
        "volmult30": EnvConfig(**base, volatile_mult=30.0),
        "m16": EnvConfig(**base, speeds=tuple(list((1.00, 1.20, 0.82, 1.45, 0.95, 1.12, 0.72, 1.30)) * 2),
                         n_volatile=4),
        "baseline": EnvConfig(**base),
    }
    for key, cfg in variants.items():
        jobs += [(i, cfg, s, pols, key) for i in INSTANCES for s in TEST_SEEDS]
    out = save_config("sensitivity", {"seeds": TEST_SEEDS, "policies": pols,
                                      "variants": {k: asdict(v) for k, v in variants.items()},
                                      "n_jobs": len(jobs)})
    run_grid(jobs, out)


def shock():
    pols = ["EFT", "ECT-Oracle", "ECT-Online", "ECT-OnlinePooled", "AR-v4",
            f"RT-Oracle-b{B:g}", f"RT-Online-b{B:g}", f"RT-OnlinePooled-b{B:g}"]
    jobs = []
    for r in (0.03, 0.1):
        cfg = EnvConfig(rho=r, coupling="uniform", shock=(3, 0.3, 5.0))
        jobs += [(i, cfg, s, pols, "") for i in INSTANCES for s in TEST_SEEDS]
    out = save_config("shock", {"seeds": TEST_SEEDS, "policies": pols, "rhos": (0.03, 0.1),
                                "shock": "3 random VMs down for 5 time units at 0.3*M0", "n_jobs": len(jobs)})
    run_grid(jobs, out)


if __name__ == "__main__":
    t0 = time.time()
    {"tuning": tuning, "tuning_modes": tuning_modes, "tuning_rt_online": tuning_rt_online, "choose_beta": choose_beta, "tuning_cd": tuning_cd, "e5": e5, "e6": e6, "tuning_age": tuning_age, "main": main, "history": history, "sensitivity": sensitivity,
     "shock": shock}[sys.argv[1]]()
    print(f"done in {time.time() - t0:.1f}s")
