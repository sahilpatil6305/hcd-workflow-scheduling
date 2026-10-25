"""Pre-registered analysis (docs/RESEARCH_DESIGN.md §8, §10).  Usage: python scripts/analyze.py
Reads results/<exp>/raw.csv, writes results/analysis/*.csv and figures/*.png|pdf.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "analysis")
FIG = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
N_BOOT = 10_000
RNG_SEED = 20261024
RHOS = (0.01, 0.03, 0.1, 0.3)
COUPLINGS = ("uniform", "fast-volatile", "random-volatile", "slow-volatile")
CELL = ["cfg_rho", "cfg_coupling"]
REP = ["instance", "seed"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out


def log_ratios(df, cell_cols, base="EFT"):
    w = df.pivot_table(index=cell_cols + REP, columns="policy", values="makespan")
    return np.log(w.div(w[base], axis=0)).drop(columns=[base])


def strat_boot_idx(instances, rng, n_boot):
    """Bootstrap index matrix resampling seeds within each instance (stratified)."""
    groups = [np.flatnonzero(instances == g) for g in np.unique(instances)]
    out = np.empty((n_boot, len(instances)), dtype=np.int64)
    pos = 0
    for g in groups:
        out[:, pos:pos + len(g)] = rng.choice(g, size=(n_boot, len(g)), replace=True)
        pos += len(g)
    return out


def summarise(lr, cell_cols, family=None, recovery_pairs=()):
    """Per cell and policy: GM ratio, stratified bootstrap CI, Wilcoxon p.
    recovery_pairs: (online, oracle) pairs for recovery fraction with joint bootstrap."""
    rng = np.random.default_rng(RNG_SEED)
    rows, rec_rows = [], []
    for cell, sub in lr.groupby(level=cell_cols):
        cell = cell if isinstance(cell, tuple) else (cell,)
        inst = sub.index.get_level_values("instance").to_numpy()
        B = strat_boot_idx(inst, rng, N_BOOT)
        boots = {}
        for pol in sub.columns:
            x = sub[pol].to_numpy()
            if np.isnan(x).all():
                continue
            bm = x[B].mean(axis=1)
            boots[pol] = bm
            nz = x[np.abs(x) > 1e-12]
            p = 1.0 if len(nz) == 0 else stats.wilcoxon(x, zero_method="wilcox").pvalue
            # exploratory (not pre-registered): sign-flip permutation test of the MEAN log ratio,
            # which targets the same estimand as the bootstrap CI (Wilcoxon targets a location shift).
            flips = rng.choice((-1.0, 1.0), size=(2000, len(x)))
            perm_p = (1 + np.sum(np.abs((flips * x).mean(axis=1)) >= abs(x.mean()) - 1e-15)) / 2001
            rows.append({**dict(zip(cell_cols, cell)), "policy": pol, "n": len(x), "signflip_p": perm_p,
                         "gm_ratio": np.exp(x.mean()), "ci_lo": np.exp(np.quantile(bm, .025)),
                         "ci_hi": np.exp(np.quantile(bm, .975)), "median_ratio": np.exp(np.median(x)),
                         "frac_better": (x < -1e-12).mean(), "frac_worse": (x > 1e-12).mean(),
                         "frac_identical": (np.abs(x) <= 1e-12).mean(), "wilcoxon_p": p})
        for on, orc in recovery_pairs:
            if on in boots and orc in boots:
                g_or = 1 - np.exp(sub[orc].mean()); g_on = 1 - np.exp(sub[on].mean())
                bor = 1 - np.exp(boots[orc]); bon = 1 - np.exp(boots[on])
                sig = np.quantile(boots[orc], .975) < 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    rb = bon / bor
                rec_rows.append({**dict(zip(cell_cols, cell)), "online": on, "oracle": orc,
                                 "oracle_gain": g_or, "online_gain": g_on,
                                 "oracle_gain_significant": bool(sig),
                                 "recovery": g_on / g_or if sig else np.nan,
                                 "rec_ci_lo": np.nanquantile(rb, .025) if sig else np.nan,
                                 "rec_ci_hi": np.nanquantile(rb, .975) if sig else np.nan})
    res = pd.DataFrame(rows)
    if family is not None:
        res["family"] = ""
        for name, pols in family.items():
            m = res.policy.isin(pols)
            res.loc[m, "family"] = name
            res.loc[m, "holm_p"] = holm(res.loc[m, "wilcoxon_p"].to_numpy())
            res.loc[m, "holm_signflip_p"] = holm(res.loc[m, "signflip_p"].to_numpy())
    return res, pd.DataFrame(rec_rows)


def pairwise(df, cell_cols, a, b):
    """GM ratio of policy a over policy b, per cell, with stratified bootstrap CI."""
    w = df.pivot_table(index=cell_cols + REP, columns="policy", values="makespan")
    x = np.log(w[a] / w[b])
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    for cell, s in x.groupby(level=cell_cols):
        cell = cell if isinstance(cell, tuple) else (cell,)
        inst = s.index.get_level_values("instance").to_numpy()
        bm = s.to_numpy()[strat_boot_idx(inst, rng, N_BOOT)].mean(axis=1)
        v = s.to_numpy()
        p = 1.0 if np.allclose(v, 0) else stats.wilcoxon(v).pvalue
        rows.append({**dict(zip(cell_cols, cell)), "a": a, "b": b, "gm_ratio_a_over_b": np.exp(v.mean()),
                     "ci_lo": np.exp(np.quantile(bm, .025)), "ci_hi": np.exp(np.quantile(bm, .975)),
                     "wilcoxon_p": p})
    return pd.DataFrame(rows)


def deadline_table(df, cell_cols, deltas=(1.1, 1.25, 1.5)):
    rows = []
    for key, s in df.groupby(list(cell_cols) + ["policy"]):
        cell, pol = key[:-1], key[-1]
        for d in deltas:
            rows.append({**dict(zip(cell_cols, cell)), "policy": pol, "delta": d,
                         "p_meet": float((s.makespan <= d * s.m0).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
COLORS = {"ECT-Oracle": "#2a78d6", "ECT-Online": "#2a78d6", "RT-Oracle-b0.5": "#eb6834",
          "RT-Online-b0.5": "#eb6834", "AR-v4": "#1baf7a", "ECTR-Oracle": "#eda100"}
STYLE = {"ECT-Oracle": ("-", "o"), "ECT-Online": ("--", "o"), "RT-Oracle-b0.5": ("-", "s"),
         "RT-Online-b0.5": ("--", "s"), "AR-v4": (":", "^"), "ECTR-Oracle": ("-", "D")}
LABEL = {"ECT-Oracle": "ECT, known rates", "ECT-Online": "ECT, online estimate",
         "RT-Oracle-b0.5": "RT (β=0.5), known rates", "RT-Online-b0.5": "RT (β=0.5), online estimate",
         "AR-v4": "AR-HEFT v4 rule (online)", "ECTR-Oracle": "ECT retry-in-place, known rates"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False,
                         "font.family": "DejaVu Sans", "savefig.dpi": 300})
    return plt


def fig_main(res):
    plt = _mpl()
    pols = ["ECT-Oracle", "ECT-Online", "RT-Oracle-b0.5", "RT-Online-b0.5", "AR-v4"]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.35), sharey=True)
    for ax, c in zip(axes, COUPLINGS):
        ax.axhline(0, color="#52514e", lw=0.8)
        for k, p in enumerate(pols):
            s = res[(res.cfg_coupling == c) & (res.policy == p)].sort_values("cfg_rho")
            if s.empty:
                continue
            x = np.arange(len(s)) + (k - 2) * 0.07
            y = (s.gm_ratio - 1) * 100
            ls, mk = STYLE[p]
            ax.errorbar(x, y, yerr=[y - (s.ci_lo - 1) * 100, (s.ci_hi - 1) * 100 - y], color=COLORS[p],
                        ls=ls, marker=mk, ms=3.5, lw=1.4, elinewidth=0.8, capsize=0, label=LABEL[p])
        ax.set_xticks(range(4), [f"{r:g}" for r in RHOS])
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("exposure ρ")
    axes[0].set_ylabel("makespan vs EFT (%)\n(geometric mean, 95% CI)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.3), ncol=3, fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_main_gm_ratio.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_distribution(lr):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2), sharey=True)
    for ax, (rho, c) in zip(axes, [(0.1, "fast-volatile"), (0.3, "uniform")]):
        sub = lr.xs((rho, c), level=CELL)
        for p in ["ECT-Oracle", "RT-Oracle-b0.5"]:
            x = np.sort(np.exp(sub[p].to_numpy()))
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLORS[p], lw=1.6, label=LABEL[p])
        ax.axvline(1, color="#52514e", lw=0.8)
        ax.set_xscale("log")
        ticks = [0.25, 0.5, 1, 2, 4, 8]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(0.2, 12)
        ax.set_title(f"ρ={rho:g}, {c}", fontsize=8)
        ax.set_xlabel("per-replicate makespan ratio vs EFT (log scale)")
    axes[0].set_ylabel("cumulative fraction")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_ecdf.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_history(hres):
    plt = _mpl()
    pols = [("ECT-Online", "ECT-Hist", "ECT-HistDisc", "ECT-Oracle"),
            ("RT-Online-b0.5", "RT-Hist-b0.5", "RT-HistDisc-b0.5", "RT-Oracle-b0.5")]
    names = ["online only", "history", "history + forgetting", "known rates"]
    fills = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    conds = [(5, False), (50, False), (5, True), (50, True)]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.6), sharey="row")
    for r_i, fam in enumerate(pols):
        for c_i, rho in enumerate((0.1, 0.3)):
            ax = axes[r_i, c_i]
            ax.axhline(0, color="#52514e", lw=0.8)
            for k, p in enumerate(fam):
                ys, lo, hi = [], [], []
                for kk, sh in conds:
                    s = hres[(hres.cfg_history_k == kk) & (hres.cfg_regime_shift == sh) &
                             (hres.cfg_rho == rho) & (hres.policy == p)].iloc[0]
                    ys.append((s.gm_ratio - 1) * 100); lo.append((s.ci_lo - 1) * 100); hi.append((s.ci_hi - 1) * 100)
                x = np.arange(4) + (k - 1.5) * 0.19
                ax.bar(x, ys, width=0.17, color=fills[k], label=names[k], edgecolor="white", linewidth=0.5)
                ax.errorbar(x, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], fmt="none", ecolor="#0b0b0b", elinewidth=0.7)
            ax.set_xticks(range(4), ["k=5\nstationary", "k=50\nstationary", "k=5\nshift", "k=50\nshift"])
            ax.set_title(f"{'ECT' if r_i == 0 else 'RT (β=0.5)'}, ρ={rho:g}", fontsize=8)
            if c_i == 0:
                ax.set_ylabel("makespan vs EFT (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_history.{ext}"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    tables = {}
    # ---- E1 main
    df = pd.read_csv(os.path.join(ROOT, "results", "main", "raw.csv"))
    assert df.work_conserved.all()
    lr = log_ratios(df, CELL)
    fam = {"primary": ["ECT-Oracle", "ECT-Online"], "secondary_RT": ["RT-Oracle-b0.5", "RT-Online-b0.5"]}
    res, rec = summarise(lr, CELL, family=fam,
                         recovery_pairs=[("ECT-Online", "ECT-Oracle"), ("RT-Online-b0.5", "RT-Oracle-b0.5"),
                                         ("ECT-OnlinePooled", "ECT-Oracle"), ("RT-OnlinePooled-b0.5", "RT-Oracle-b0.5")])
    res.to_csv(os.path.join(OUT, "E1_gm_ratio_vs_EFT.csv"), index=False)
    rec.to_csv(os.path.join(OUT, "E1_recovery.csv"), index=False)
    pw = pd.concat([pairwise(df, CELL, "ECT-Oracle", "RT-Oracle-b0.5"),
                    pairwise(df, CELL, "ECT-Oracle", "ECT-OraclePooled"),
                    pairwise(df, CELL, "RT-Oracle-b0.5", "RT-OraclePooled-b0.5"),
                    pairwise(df, CELL, "ECTR-Oracle", "ECT-Oracle")])
    pw.to_csv(os.path.join(OUT, "E1_pairwise.csv"), index=False)
    app = lr.groupby([lr.index.get_level_values("cfg_rho"), lr.index.get_level_values("cfg_coupling"),
                      lr.index.get_level_values("instance").str.split("-").str[0]]).mean()
    app.index.names = ["cfg_rho", "cfg_coupling", "app"]
    (np.exp(app)).reset_index().to_csv(os.path.join(OUT, "E1_per_app_gm_ratio.csv"), index=False)
    deadline_table(df, CELL).to_csv(os.path.join(OUT, "E1_deadline_meeting.csv"), index=False)
    ov = df.groupby("policy").apply(lambda s: pd.Series({
        "us_per_decision_mean": 1e6 * s.sched_time_s.sum() / s.decisions.sum(),
        "decisions_mean": s.decisions.mean(), "lost_work_mean": s.lost_work.mean(),
        "interruptions_mean": s.interruptions.mean()}), include_groups=False)
    ov.to_csv(os.path.join(OUT, "E1_overhead_and_loss.csv"))
    lw = df.groupby(CELL + ["policy"])[["lost_work", "interruptions", "utilization"]].mean().reset_index()
    lw.to_csv(os.path.join(OUT, "E1_lost_work_by_cell.csv"), index=False)
    hz = df[df.policy == "EFT"].groupby(CELL).apply(lambda s: (s.horizon > 30 * s.m0 + 50 + 1e-6).mean(),
                                                     include_groups=False)
    hz.rename("frac_horizon_extended").reset_index().to_csv(os.path.join(OUT, "E1_horizon_extensions.csv"), index=False)
    fig_main(res)
    fig_distribution(lr)

    # ---- E2 history
    hp = os.path.join(ROOT, "results", "history", "raw.csv")
    if os.path.exists(hp):
        h = pd.read_csv(hp)
        hc = ["cfg_history_k", "cfg_regime_shift", "cfg_rho"]
        hl = log_ratios(h, hc)
        hres, hrec = summarise(hl, hc, recovery_pairs=[("ECT-Online", "ECT-Oracle"), ("ECT-Hist", "ECT-Oracle"),
                                                        ("ECT-HistDisc", "ECT-Oracle"), ("RT-Online-b0.5", "RT-Oracle-b0.5"),
                                                        ("RT-Hist-b0.5", "RT-Oracle-b0.5"), ("RT-HistDisc-b0.5", "RT-Oracle-b0.5")])
        hres.to_csv(os.path.join(OUT, "E2_gm_ratio_vs_EFT.csv"), index=False)
        hrec.to_csv(os.path.join(OUT, "E2_recovery.csv"), index=False)
        pd.concat([pairwise(h, hc, "ECT-HistDisc", "ECT-Hist"), pairwise(h, hc, "RT-HistDisc-b0.5", "RT-Hist-b0.5"),
                   pairwise(h, hc, "ECT-Hist", "ECT-Online"), pairwise(h, hc, "RT-Hist-b0.5", "RT-Online-b0.5")]
                  ).to_csv(os.path.join(OUT, "E2_pairwise.csv"), index=False)
        fig_history(hres)

    # ---- E3 sensitivity
    sp = os.path.join(ROOT, "results", "sensitivity", "raw.csv")
    if os.path.exists(sp):
        s = pd.read_csv(sp)
        # variant key: reconstruct from config columns
        s["variant"] = np.select(
            [s.cfg_repair == 0.25, s.cfg_repair == 4.0, s.cfg_volatile_mult == 3.0, s.cfg_volatile_mult == 30.0,
             s.cfg_n_volatile == 4], ["repair0.25", "repair4", "volmult3", "volmult30", "m16"], "baseline")
        sl = log_ratios(s, ["variant"])
        sres, _ = summarise(sl, ["variant"])
        sres.to_csv(os.path.join(OUT, "E3_sensitivity.csv"), index=False)

    # ---- E4 shock
    kp = os.path.join(ROOT, "results", "shock", "raw.csv")
    if os.path.exists(kp):
        k = pd.read_csv(kp)
        kl = log_ratios(k, ["cfg_rho"])
        kres, _ = summarise(kl, ["cfg_rho"])
        kres.to_csv(os.path.join(OUT, "E4_shock.csv"), index=False)
    print("analysis written to", OUT)


if __name__ == "__main__":
    main()
