"""v5.1: regret summaries (results/analysis/E5_regret.csv, E6_regret.csv) and LaTeX tables tab_e5/tab_e6.
Run after scripts/run_experiments.py e5/e6. Then re-run scripts/analyze_v51.py to draw fig_regret.
Usage: python scripts/make_tables_v51.py"""
import json, os
import numpy as np, pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(R, "results", "analysis")
T = os.path.join(R, "manuscript", "tables")
E5 = os.path.join(R, "results", "e5_stale_history", "raw.csv")
E6 = os.path.join(R, "results", "e6_bursty", "raw.csv")
BASE = ["RT-Online-b0.5", "RT-Hist-b0.5", "RT-HistDisc-b0.5", "RT-HistCD-b0.5-a0.05"]


def load(p):
    d = pd.read_csv(p)
    if "cfg_uptime_shape" not in d.columns:
        d["cfg_uptime_shape"] = 1.0
    d["cfg_uptime_shape"] = d.cfg_uptime_shape.fillna(1.0)
    return d


def lr(d, C):
    w = d.pivot_table(index=C + ["instance", "seed"], columns="policy", values="makespan")
    return np.log(w.div(w["EFT"], axis=0))


def regret():
    out = {}
    for name, p, C, pols in [("E5", E5, ["cfg_coupling", "cfg_rho", "cfg_history_k", "cfg_regime_shift"], BASE),
                             ("E6", E6, ["cfg_uptime_shape", "cfg_coupling", "cfg_rho", "cfg_history_k", "cfg_regime_shift"],
                              BASE + ["RT-HistCDQ-b0.5-a0.05"])]:
        L = lr(load(p), C)
        cell = (np.exp(L.groupby(level=C).mean()[pols]) - 1) * 100
        reg = cell.sub(cell.min(axis=1), axis=0)
        res = pd.DataFrame({"max_regret": reg.max(), "mean_regret": reg.mean(),
                            "cells_best": (reg < 1e-12).sum(), "n_cells": len(reg)})
        if name == "E6":
            for shp in (1.0, 0.7, 0.5):
                res[f"max_regret_k{shp}"] = reg.xs(shp, level="cfg_uptime_shape").max()
        res.to_csv(os.path.join(A, f"{name}_regret.csv"))
        out[name] = res.round(3).to_dict()
    json.dump(out, open(os.path.join(A, "regret_summary.json"), "w"), indent=1)


def fmt(v, best):
    s = f"{v:+.2f}"
    return r"\textbf{" + s + "}" if best else s


def tables():
    pols5 = BASE + ["RT-Oracle-b0.5"]
    d5 = load(E5)
    t5 = (np.exp(lr(d5, ["cfg_coupling", "cfg_regime_shift", "cfg_rho", "cfg_history_k"])
                 .groupby(level=["cfg_coupling", "cfg_regime_shift"]).mean()[pols5]) - 1) * 100
    allv = (np.exp(lr(d5, ["cfg_coupling", "cfg_regime_shift", "cfg_rho", "cfg_history_k"]).mean()[pols5]) - 1) * 100
    L = [r"\begin{table}[t]\centering\scriptsize",
         r"\caption{E5: using monitoring history with and without change detection (RT, $\beta=0.5$, exponential up-times). "
         r"Geometric-mean makespan change vs.\ \EFT{} (\%), pooled over $\rho\in\{0.03,0.1,0.3\}$, $k\in\{5,50\}$ and 14 workflows "
         r"(8{,}400 paired replicates per entry). On = online only, H = history, HF = history with forgetting, "
         r"HCD = history with change detection (new), K = known rates. Bold: best non-oracle rule.}\label{tab:e5}",
         r"\setlength{\tabcolsep}{3pt}\begin{tabular}{llrrrrr}", r"\toprule",
         r"rates & regime & On & H & HF & HCD & K\\", r"\midrule"]
    for (c, sh), row in t5.iterrows():
        v = row.values; b = int(np.argmin(v[:4]))
        L.append(f"{'fast-vol.' if c == 'fast-volatile' else 'random-vol.'} & {'shift' if sh else 'stationary'} & "
                 + " & ".join(fmt(x, i == b) for i, x in enumerate(v)) + r"\\")
    v = allv.values; b = int(np.argmin(v[:4]))
    L.append(r"\midrule all & both & " + " & ".join(fmt(x, i == b) for i, x in enumerate(v)) + r"\\")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    open(os.path.join(T, "tab_e5.tex"), "w").write("\n".join(L))

    pols6 = BASE + ["RT-HistCDQ-b0.5-a0.05", "RT-Oracle-b0.5"]
    d6 = load(E6)
    t6 = (np.exp(lr(d6, ["cfg_uptime_shape", "cfg_regime_shift", "cfg_coupling", "cfg_rho", "cfg_history_k"])
                 .groupby(level=["cfg_uptime_shape", "cfg_regime_shift"]).mean()[pols6]) - 1) * 100
    L = [r"\begin{table}[t]\centering\scriptsize",
         r"\caption{E6: bursty (Weibull) up-times with the same mean. Makespan change vs.\ \EFT{} (\%), pooled over both couplings, "
         r"$\rho\in\{0.03,0.1\}$, $k\in\{5,50\}$ and 14 workflows (11{,}200 replicates per entry). HCDQ = HCD with dispersion correction.}\label{tab:e6}",
         r"\setlength{\tabcolsep}{2.5pt}\begin{tabular}{llrrrrrr}", r"\toprule",
         r"up-times & regime & On & H & HF & HCD & HCDQ & K\\", r"\midrule"]
    for (shp, sh), row in t6.sort_index(ascending=[False, True]).iterrows():
        v = row.values; b = int(np.argmin(v[:5]))
        lab = {1.0: "exponential", 0.7: "Weibull 0.7", 0.5: "Weibull 0.5"}[shp]
        L.append(f"{lab} & {'shift' if sh else 'stationary'} & " + " & ".join(fmt(x, i == b) for i, x in enumerate(v)) + r"\\")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    open(os.path.join(T, "tab_e6.tex"), "w").write("\n".join(L))


if __name__ == "__main__":
    regret(); tables(); print("v5.1 regret + tables written")
