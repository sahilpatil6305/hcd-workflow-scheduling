"""Generate LaTeX tables and a numbers file for the manuscript from results/analysis/*.csv."""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "results", "analysis")
T = os.path.join(ROOT, "manuscript", "tables")
os.makedirs(T, exist_ok=True)
COUP = ["uniform", "fast-volatile", "random-volatile", "slow-volatile"]
SHORT = {"uniform": "uniform", "fast-volatile": "fast-vol.", "random-volatile": "random-vol.", "slow-volatile": "slow-vol."}


def pct(r, lo=None, hi=None, bold=False):
    s = f"{(r - 1) * 100:+.1f}"
    if lo is not None:
        sig = hi < 1 or lo > 1
        s = f"{s} [{(lo - 1) * 100:+.1f},{(hi - 1) * 100:+.1f}]"
        if sig and bold:
            s = r"\textbf{" + s + "}"
    return s


def t_main():
    r = pd.read_csv(os.path.join(A, "E1_gm_ratio_vs_EFT.csv"))
    cols = [("ECT-Oracle", "ECT known"), ("ECT-Online", "ECT online"), ("RT-Oracle-b0.5", "RT known"),
            ("RT-Online-b0.5", "RT online"), ("ECT-OraclePooled", "ECT pooled"), ("AR-v4", "AR-v4")]
    lines = [r"\begin{table*}[t]\centering\scriptsize",
             r"\caption{Geometric-mean makespan change vs.\ EFT (\%) with 95\% stratified-bootstrap CI; "
             r"1{,}400 paired replicates per row (14 workflows $\times$ 100 test seeds). Bold: CI excludes 0. "
             r"Negative is better. ECT/RT ``known'' use true per-VM rates; ``online'' estimates them from monitoring; "
             r"``pooled'' uses the true pool-mean rate for every VM.}\label{tab:main}",
             r"\setlength{\tabcolsep}{3pt}\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             r"$\rho$ & rates & " + " & ".join(c[1] for c in cols) + r"\\", r"\midrule"]
    for rho in (0.01, 0.03, 0.1, 0.3):
        for c in COUP:
            cells = []
            for p, _ in cols:
                x = r[(r.cfg_rho == rho) & (r.cfg_coupling == c) & (r.policy == p)].iloc[0]
                cells.append(pct(x.gm_ratio, x.ci_lo, x.ci_hi, bold=True))
            lines.append(f"{rho:g} & {SHORT[c]} & " + " & ".join(cells) + r"\\")
        if rho != 0.3:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule\end{tabular}\end{table*}"]
    open(os.path.join(T, "tab_main.tex"), "w").write("\n".join(lines))


def t_recovery():
    r = pd.read_csv(os.path.join(A, "E1_recovery.csv"))
    r = r[r.oracle_gain_significant & r.online.isin(["ECT-Online", "RT-Online-b0.5"])]
    lines = [r"\begin{table}[t]\centering\scriptsize",
             r"\caption{Fraction of the known-rate gain recovered by online estimation (no history), "
             r"in cells where the known-rate gain is significant. 95\% bootstrap CI.}\label{tab:recovery}",
             r"\begin{tabular}{llrrr}", r"\toprule", r"$\rho$ & rates & rule & known gain & recovery \\", r"\midrule"]
    for _, x in r.sort_values(["online", "cfg_rho"]).iterrows():
        rule = "ECT" if x.online.startswith("ECT") else "RT"
        lines.append(f"{x.cfg_rho:g} & {SHORT[x.cfg_coupling]} & {rule} & {x.oracle_gain * 100:.1f}\\% & "
                     f"{x.recovery:.2f} [{x.rec_ci_lo:.2f}, {x.rec_ci_hi:.2f}]\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    open(os.path.join(T, "tab_recovery.tex"), "w").write("\n".join(lines))


def t_history():
    h = pd.read_csv(os.path.join(A, "E2_gm_ratio_vs_EFT.csv"))
    pols = [("ECT-Online", "ECT-Hist", "ECT-HistDisc", "ECT-Oracle"),
            ("RT-Online-b0.5", "RT-Hist-b0.5", "RT-HistDisc-b0.5", "RT-Oracle-b0.5")]
    lines = [r"\begin{table}[t]\centering\scriptsize",
             r"\caption{Monitoring history (E2; random-volatile). Makespan change vs.\ EFT (\%). "
             r"$k$: expected pre-run outages per VM; ``shift'': volatile VMs change at $t=0$. "
             r"On = online only, H = history, HD = history with forgetting, K = known rates.}\label{tab:history}",
             r"\setlength{\tabcolsep}{2.5pt}\begin{tabular}{lll rrrr rrrr}", r"\toprule",
             r" & & & \multicolumn{4}{c}{ECT} & \multicolumn{4}{c}{RT} \\ \cmidrule(lr){4-7}\cmidrule(lr){8-11}",
             r"$\rho$ & $k$ & regime & On & H & HD & K & On & H & HD & K\\", r"\midrule"]
    for rho in (0.1, 0.3):
        for sh in (False, True):
            for k in (5, 50):
                vals = []
                for fam in pols:
                    for p in fam:
                        x = h[(h.cfg_rho == rho) & (h.cfg_history_k == k) & (h.cfg_regime_shift == sh) & (h.policy == p)].iloc[0]
                        s = pct(x.gm_ratio)
                        if x.ci_hi < 1 or x.ci_lo > 1:
                            s = r"\textbf{" + s + "}"
                        vals.append(s)
                lines.append(f"{rho:g} & {k} & {'shift' if sh else 'stat.'} & " + " & ".join(vals) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    open(os.path.join(T, "tab_history.tex"), "w").write("\n".join(lines))


def t_sens():
    s = pd.read_csv(os.path.join(A, "E3_sensitivity.csv"))
    names = {"baseline": "baseline ($R{=}1$, $\\times$10, $m{=}8$)", "repair0.25": "repair $R=0.25$",
             "repair4": "repair $R=4$", "volmult3": "volatile $\\times$3", "volmult30": "volatile $\\times$30",
             "m16": "$m=16$ (4 volatile)"}
    cols = [("ECT-Oracle", "ECT K"), ("ECT-Online", "ECT On"), ("ECT-Online-prior0.25", "prior/4"),
            ("ECT-Online-prior4", "prior$\\times$4"), ("RT-Oracle-b0.5", "RT K"), ("RT-Online-b0.5", "RT On"),
            ("RT-Online-b0.5-prior0.25", "prior/4"), ("RT-Online-b0.5-prior4", "prior$\\times$4")]
    lines = [r"\begin{table}[t]\centering\scriptsize",
             r"\caption{Sensitivity (E3; $\rho=0.1$, fast-volatile). Makespan change vs.\ EFT (\%). "
             r"All entries have 95\% CIs excluding 0 unless marked $^\dagger$.}\label{tab:sens}",
             r"\setlength{\tabcolsep}{2pt}\resizebox{\columnwidth}{!}{\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(c[1] for c in cols) + r"\\", r"\midrule"]
    for v in ["baseline", "repair0.25", "repair4", "volmult3", "volmult30", "m16"]:
        vals = []
        for p, _ in cols:
            x = s[(s.variant == v) & (s.policy == p)].iloc[0]
            t = pct(x.gm_ratio)
            if not (x.ci_hi < 1 or x.ci_lo > 1):
                t += r"$^\dagger$"
            vals.append(t)
        lines.append(names[v] + " & " + " & ".join(vals) + r"\\")
    lines += [r"\bottomrule\end{tabular}}\end{table}"]
    open(os.path.join(T, "tab_sens.tex"), "w").write("\n".join(lines))


def t_apps():
    a = pd.read_csv(os.path.join(A, "E1_per_app_gm_ratio.csv"))
    cols = [("ECT-Oracle", "ECT K"), ("ECT-Online", "ECT On"), ("RT-Oracle-b0.5", "RT K"), ("RT-Online-b0.5", "RT On")]
    lines = [r"\begin{table}[t]\centering\scriptsize",
             r"\caption{Per-application geometric-mean makespan change vs.\ EFT (\%), 2 instances $\times$ 100 seeds each.}\label{tab:apps}",
             r"\setlength{\tabcolsep}{2.5pt}\begin{tabular}{l rrrr rrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{$\rho=0.1$, fast-volatile} & \multicolumn{4}{c}{$\rho=0.3$, uniform}\\"
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             "app & " + " & ".join(c[1] for c in cols) * 1 + " & " + " & ".join(c[1] for c in cols) + r"\\", r"\midrule"]
    for app in sorted(a.app.unique()):
        vals = []
        for rho, c in ((0.1, "fast-volatile"), (0.3, "uniform")):
            x = a[(a.app == app) & (a.cfg_rho == rho) & (a.cfg_coupling == c)].iloc[0]
            vals += [pct(x[p]) for p, _ in cols]
        lines.append(f"{app} & " + " & ".join(vals) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    open(os.path.join(T, "tab_apps.tex"), "w").write("\n".join(lines))


def numbers():
    """Key numbers quoted in the text, so they can be checked against the CSVs."""
    r = pd.read_csv(os.path.join(A, "E1_gm_ratio_vs_EFT.csv"))
    g = lambda rho, c, p: float((r[(r.cfg_rho == rho) & (r.cfg_coupling == c) & (r.policy == p)].iloc[0].gm_ratio - 1) * 100)
    out = {f"{p}|{rho}|{c}": round(g(rho, c, p), 2) for p in r.policy.unique() for rho in (0.01, 0.03, 0.1, 0.3) for c in COUP}
    fam = r[r.family == "primary"]
    out["primary_sig_wilcoxon_holm"] = int((fam.holm_p < 0.05).sum())
    out["primary_sig_wilcoxon_better"] = int(((fam.holm_p < 0.05) & (fam.gm_ratio < 1)).sum())
    out["primary_sig_signflip_holm"] = int((fam.holm_signflip_p < 0.05).sum())
    out["primary_sig_signflip_better"] = int(((fam.holm_signflip_p < 0.05) & (fam.gm_ratio < 1)).sum())
    sec = r[r.family == "secondary_RT"]
    out["rt_sig_signflip_holm"] = int((sec.holm_signflip_p < 0.05).sum())
    out["rt_sig_signflip_better"] = int(((sec.holm_signflip_p < 0.05) & (sec.gm_ratio < 1)).sum())
    out["rt_max_loss_pct"] = round(float((r[r.policy.isin(["RT-Oracle-b0.5", "RT-Online-b0.5"])].gm_ratio.max() - 1) * 100), 2)
    out["ect_max_loss_pct"] = round(float((r[r.policy.isin(["ECT-Oracle", "ECT-Online"])].gm_ratio.max() - 1) * 100), 2)
    ov = pd.read_csv(os.path.join(A, "E1_overhead_and_loss.csv")).set_index("policy")
    out["us_per_decision"] = ov.us_per_decision_mean.round(2).to_dict()
    lw = pd.read_csv(os.path.join(A, "E1_lost_work_by_cell.csv"))
    out["lost_work_0.3_uniform"] = lw[(lw.cfg_rho == 0.3) & (lw.cfg_coupling == "uniform")].set_index("policy").lost_work.round(1).to_dict()
    hz = pd.read_csv(os.path.join(A, "E1_horizon_extensions.csv"))
    out["horizon_extended_frac_max"] = float(hz.frac_horizon_extended.max())
    json.dump(out, open(os.path.join(ROOT, "manuscript", "numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    for f in (t_main, t_recovery, t_history, t_sens, t_apps, numbers):
        f()
    print("tables written to", T)
