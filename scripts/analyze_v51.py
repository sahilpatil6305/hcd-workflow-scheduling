"""v5.1 analysis: E5 (stale-history detection) and E6 (bursty failures).
Usage: python scripts/analyze_v51.py   -> results/analysis/E5_*, E6_*, figures/fig_e5*, fig_e6*
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import FIG, OUT, ROOT, _mpl, holm, log_ratios, pairwise, strat_boot_idx, summarise  # noqa: E402

RNG = np.random.default_rng(20261025)
B = 2000


def signflip_p(x, rng):
    flips = rng.choice((-1.0, 1.0), size=(B, len(x)))
    return (1 + np.sum(np.abs((flips * x).mean(axis=1)) >= abs(x.mean()) - 1e-15)) / (B + 1)


def contrasts(df, cells, pairs):
    w = df.pivot_table(index=cells + ["instance", "seed"], columns="policy", values="makespan")
    rows = []
    for a, b in pairs:
        x = np.log(w[a] / w[b])
        for cell, s in x.groupby(level=cells):
            v = s.to_numpy()
            inst = s.index.get_level_values("instance").to_numpy()
            bm = v[strat_boot_idx(inst, RNG, 10_000)].mean(axis=1)
            from scipy import stats as _st
            rows.append({**dict(zip(cells, cell)), "a": a, "b": b, "pct": (np.exp(v.mean()) - 1) * 100,
                         "t_p": float(_st.ttest_1samp(v, 0.0).pvalue) if np.ptp(v) > 0 else 1.0,
                         "lo": (np.exp(np.quantile(bm, .025)) - 1) * 100, "hi": (np.exp(np.quantile(bm, .975)) - 1) * 100,
                         "signflip_p": signflip_p(v, RNG)})
    out = pd.DataFrame(rows)
    # Holm on paired t-test p-values (n = 1,400 log-ratios per cell). The sign-flip p-value
    # (B = 2,000) has resolution 5e-4, too coarse for Holm over many contrasts; kept for reference.
    out["holm_p"] = holm(out.t_p.to_numpy())
    return out


def fig_e5(res, cd):
    plt = _mpl()
    pols = [("RT-Online-b0.5", "online only", "#2a78d6", "o"), ("RT-Hist-b0.5", "history", "#eb6834", "s"),
            ("RT-HistDisc-b0.5", "history + forgetting", "#eda100", "D"),
            (cd, "history + change detection (new)", "#1baf7a", "^"), ("RT-Oracle-b0.5", "known rates", "#52514e", "x")]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.4), sharey=True)
    combos = [(c, sh) for c in ("fast-volatile", "random-volatile") for sh in (False, True)]
    xt = [(r, k) for r in (0.03, 0.1, 0.3) for k in (5, 50)]
    for ax, (c, sh) in zip(axes, combos):
        ax.axhline(0, color="#52514e", lw=0.8)
        for n, (p, lab, col, mk) in enumerate(pols):
            ys, lo, hi = [], [], []
            for r, k in xt:
                s = res[(res.cfg_coupling == c) & (res.cfg_rho == r) & (res.cfg_history_k == k)
                        & (res.cfg_regime_shift == sh) & (res.policy == p)].iloc[0]
                ys.append((s.gm_ratio - 1) * 100); lo.append((s.ci_lo - 1) * 100); hi.append((s.ci_hi - 1) * 100)
            x = np.arange(len(xt)) + (n - 2) * 0.12
            ys = np.array(ys)
            ax.errorbar(x, ys, yerr=[ys - lo, np.array(hi) - ys], fmt=mk, color=col, ms=3.5, elinewidth=0.7,
                        lw=0, label=lab, mfc=col if mk != "x" else None)
        ax.set_xticks(range(len(xt)), [f"{r:g}\nk={k}" for r, k in xt], fontsize=6.5)
        ax.set_title(f"{c}, {'regime shift' if sh else 'stationary'}", fontsize=7.5)
    axes[0].set_ylabel("makespan vs EFT (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_e5_change_detection.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_e6(res, cd, cdq):
    plt = _mpl()
    pols = [("RT-Online-b0.5", "online only", "#2a78d6", "o"), ("RT-Hist-b0.5", "history", "#eb6834", "s"),
            (cd, "change detection (new)", "#1baf7a", "^"), (cdq, "+ dispersion corr.", "#6250d6", "v"),
            ("RT-Oracle-b0.5", "known rates", "#52514e", "x")]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True)
    shapes = (1.0, 0.7, 0.5)
    for ax, sh in zip(axes, (False, True)):
        ax.axhline(0, color="#52514e", lw=0.8)
        sub = res[(res.cfg_coupling == "fast-volatile") & (res.cfg_rho == 0.1) & (res.cfg_regime_shift == sh)]
        for n, (p, lab, col, mk) in enumerate(pols):
            ys, lo, hi = [], [], []
            for shp in shapes:
                s = sub[(sub.cfg_uptime_shape == shp) & (sub.policy == p) & (sub.cfg_history_k == 50)].iloc[0]
                ys.append((s.gm_ratio - 1) * 100); lo.append((s.ci_lo - 1) * 100); hi.append((s.ci_hi - 1) * 100)
            ys = np.array(ys)
            x = np.arange(3) + (n - 2) * 0.12
            ax.errorbar(x, ys, yerr=[ys - lo, np.array(hi) - ys], fmt=mk, color=col, ms=3.5, elinewidth=0.7, lw=0, label=lab)
        ax.set_xticks(range(3), ["exponential", "Weibull k=0.7", "Weibull k=0.5"])
        ax.set_title(f"fast-volatile, ρ=0.1, k=50, {'regime shift' if sh else 'stationary'}", fontsize=7.5)
    axes[0].set_ylabel("makespan vs EFT (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_e6_bursty.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_regret():
    plt = _mpl()
    r5 = pd.read_csv(os.path.join(OUT, "E5_regret.csv"), index_col=0)
    r6 = pd.read_csv(os.path.join(OUT, "E6_regret.csv"), index_col=0)
    pols = [("RT-Online-b0.5", "online only"), ("RT-Hist-b0.5", "history"),
            ("RT-HistDisc-b0.5", "history +\nforgetting"), ("RT-HistCD-b0.5-a0.05", "history +\nchange detection")]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2), sharey=True)
    for ax, (r, title) in zip(axes, [(r5, "E5: exponential failures (24 conditions)"),
                                     (r6, "E6: incl. bursty Weibull failures (48 conditions)")]):
        y = np.arange(len(pols))
        mx = [r.loc[p, "max_regret"] for p, _ in pols]
        mn = [r.loc[p, "mean_regret"] for p, _ in pols]
        cols = ["#b9b8b2"] * 3 + ["#1baf7a"]
        ax.barh(y + 0.18, mx, height=0.34, color=cols, edgecolor="white", linewidth=0.5)
        ax.barh(y - 0.18, mn, height=0.34, color=cols, alpha=0.55, edgecolor="white", linewidth=0.5)
        for yi, a, b in zip(y, mx, mn):
            ax.text(a + 0.05, yi + 0.18, f"worst {a:.2f}", va="center", fontsize=6.5, color="#0b0b0b")
            ax.text(b + 0.05, yi - 0.18, f"avg {b:.2f}", va="center", fontsize=6.5, color="#52514e")
        ax.set_yticks(y, [l for _, l in pols])
        ax.set_title(title, fontsize=7.5)
        ax.set_xlim(0, 4.6)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("regret = gap to the best rule in each condition (percentage points of makespan); lower is better",
                  fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_regret.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    summary = {}
    # ---------------- E5
    d = pd.read_csv(os.path.join(ROOT, "results", "e5_stale_history", "raw.csv"))
    assert d.work_conserved.all()
    C = ["cfg_coupling", "cfg_rho", "cfg_history_k", "cfg_regime_shift"]
    cd = [p for p in d.policy.unique() if p.startswith("RT-HistCD")][0]
    lr = log_ratios(d, C)
    res, rec = summarise(lr, C, recovery_pairs=[(p, "RT-Oracle-b0.5") for p in
                                                ("RT-Online-b0.5", "RT-Hist-b0.5", "RT-HistDisc-b0.5", cd)])
    res.to_csv(os.path.join(OUT, "E5_gm_ratio_vs_EFT.csv"), index=False)
    rec.to_csv(os.path.join(OUT, "E5_recovery.csv"), index=False)
    con = contrasts(d, C, [(cd, "RT-Hist-b0.5"), (cd, "RT-Online-b0.5"), (cd, "RT-HistDisc-b0.5")])
    con.to_csv(os.path.join(OUT, "E5_contrasts.csv"), index=False)
    by = (np.exp(lr.groupby(level="cfg_regime_shift").mean()) - 1) * 100
    by.to_csv(os.path.join(OUT, "E5_mean_by_shift.csv"))
    both = (np.exp(lr.mean()) - 1) * 100
    summary["E5_overall_pct"] = both.round(3).to_dict()
    summary["E5_by_shift_pct"] = {str(k): v for k, v in by.round(3).T.to_dict().items()}
    for (a, b), g in con.groupby(["a", "b"]):
        for sh, gg in g.groupby("cfg_regime_shift"):
            summary[f"E5|{a}|vs|{b}|shift={sh}"] = {
                "n_cells": len(gg), "better_sig": int(((gg.holm_p < .05) & (gg.pct < 0)).sum()),
                "worse_sig": int(((gg.holm_p < .05) & (gg.pct > 0)).sum()),
                "ci_excl0_better": int((gg.hi < 0).sum()), "ci_excl0_worse": int((gg.lo > 0).sum()),
                "min_pct": round(gg.pct.min(), 2), "max_pct": round(gg.pct.max(), 2)}
    fig_e5(res, cd)
    # ECT descriptive
    ect = res[res.policy.isin(["ECT-Hist", [p for p in d.policy.unique() if p.startswith("ECT-HistCD")][0]])]
    ect.to_csv(os.path.join(OUT, "E5_ect_descriptive.csv"), index=False)
    ov = d.groupby("policy").apply(lambda s: 1e6 * s.sched_time_s.sum() / s.decisions.sum(), include_groups=False)
    summary["E5_us_per_decision"] = ov.round(2).to_dict()

    # ---------------- E6
    p6 = os.path.join(ROOT, "results", "e6_bursty", "raw.csv")
    if os.path.exists(p6):
        e = pd.read_csv(p6)
        assert e.work_conserved.all()
        if "cfg_uptime_shape" not in e.columns:
            e["cfg_uptime_shape"] = 1.0
        e["cfg_uptime_shape"] = e["cfg_uptime_shape"].fillna(1.0)
        C6 = ["cfg_uptime_shape"] + C
        cd6 = [p for p in e.policy.unique() if p.startswith("RT-HistCD-")][0]
        cdq = [p for p in e.policy.unique() if p.startswith("RT-HistCDQ")][0]
        lr6 = log_ratios(e, C6)
        res6, _ = summarise(lr6, C6)
        res6.to_csv(os.path.join(OUT, "E6_gm_ratio_vs_EFT.csv"), index=False)
        con6 = contrasts(e, C6, [(cdq, cd6), (cdq, "RT-Hist-b0.5"), (cd6, "RT-Hist-b0.5"), (cdq, "RT-Online-b0.5")])
        con6.to_csv(os.path.join(OUT, "E6_contrasts.csv"), index=False)
        agg = (np.exp(lr6.groupby(level=["cfg_uptime_shape", "cfg_regime_shift"]).mean()) - 1) * 100
        agg.to_csv(os.path.join(OUT, "E6_mean_by_shape_shift.csv"))
        summary["E6_by_shape_shift_pct"] = {f"{k[0]}|{k[1]}": v for k, v in agg.round(3).T.to_dict().items()}
        for (a, b, shp, sh), g in con6.groupby(["a", "b", "cfg_uptime_shape", "cfg_regime_shift"]):
            summary[f"E6|{a}|vs|{b}|shape={shp}|shift={sh}"] = {
                "n_cells": len(g), "better_sig": int(((g.holm_p < .05) & (g.pct < 0)).sum()),
                "worse_sig": int(((g.holm_p < .05) & (g.pct > 0)).sum()),
                "min_pct": round(g.pct.min(), 2), "max_pct": round(g.pct.max(), 2), "mean_pct": round(g.pct.mean(), 2)}
        fig_e6(res6, cd6, cdq)
    if os.path.exists(os.path.join(OUT, "E6_regret.csv")):
        fig_regret()
    with open(os.path.join(OUT, "v51_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("v5.1 analysis written")


if __name__ == "__main__":
    main()
