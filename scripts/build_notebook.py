"""Build and execute the v5 notebook without Jupyter (not installed in the build environment).

Each code cell is executed in order in one namespace; stdout is captured into a "stream"
output and figures saved by the cell are embedded as PNG outputs. The result is a valid
nbformat-4 notebook whose outputs were produced by actually running the code.
Usage: python scripts/build_notebook.py
"""
import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "FAWS_v5_Research_Notebook.ipynb")

CELLS = [
    ("md", """# Failure-aware workflow placement under estimated failure rates — v5 notebook

Successor of *AR-HEFT v4*. AR-HEFT was retired after the Stage 1 audit (`docs/STAGE1_AUDIT.md`).
This notebook is a thin, executable front end to the `fawsim` package: it runs the test suite,
demonstrates one paired replicate, and reproduces every table and figure in the manuscript
from the saved raw results. The full experiments are run by `scripts/run_experiments.py`
(about 10 minutes on 2 cores); they are not re-run here.

**Research question.** When per-VM failure rates must be estimated from monitoring, how much of the
makespan benefit of failure-aware placement (with known rates) can an online scheduler recover, and
in which regimes is failure-aware placement worth using at all?"""),
    ("code", """import os, sys, subprocess, json
ROOT = os.path.abspath('.')
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import fawsim
print('fawsim', fawsim.__version__, '| numpy', np.__version__, '| pandas', pd.__version__)"""),
    ("md", "## 1. Validation tests (must pass)\nAnalytical (closed-form vs Monte Carlo), invariant, information-access and regression tests."),
    ("code", """r = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_fawsim'], capture_output=True, text=True, cwd=ROOT)
print(r.stderr.strip().splitlines()[-3:])
assert r.returncode == 0, r.stderr"""),
    ("md", "## 2. One paired replicate\nAll policies see the same workflow and the same outage trace; only the placement score and the information differ. A single replicate is very noisy (here known-rate ECT is 30% *worse* than EFT, although it is 6% better on average in this cell): placement changes alter which tasks the outages hit. Conclusions below rest on 1,400 paired replicates per cell."),
    ("code", """from fawsim.environment import EnvConfig
from fawsim.experiments import run_replicate, INSTANCES
pols = ['EFT', 'ECT-Oracle', 'ECT-Online', 'RT-Oracle-b0.5', 'RT-Online-b0.5', 'AR-v4']
rows = run_replicate(INSTANCES[11], EnvConfig(rho=0.1, coupling='fast-volatile'), 1000, pols)
demo = pd.DataFrame(rows)[['policy', 'makespan', 'lost_work', 'interruptions', 'n_outage_intervals_run', 'run_volatile']]
demo['vs_EFT_%'] = (demo.makespan / demo.makespan.iloc[0] - 1) * 100
print(demo.round(3).to_string(index=False))"""),
    ("md", "## 3. Main experiment (E1): geometric-mean makespan vs EFT\n100 test seeds × 14 WfCommons instances per cell; 95% stratified-bootstrap CIs."),
    ("code", """subprocess.run([sys.executable, 'scripts/analyze.py'], check=True, cwd=ROOT)
res = pd.read_csv('results/analysis/E1_gm_ratio_vs_EFT.csv')
res['pct'] = (res.gm_ratio - 1) * 100
tab = res.pivot_table(index=['cfg_rho', 'cfg_coupling'], columns='policy', values='pct')
print(tab[['ECT-Oracle', 'ECT-Online', 'RT-Oracle-b0.5', 'RT-Online-b0.5', 'ECT-OraclePooled', 'AR-v4']].round(2).to_string())"""),
    ("code", "FIGS = ['figures/fig_main_gm_ratio.png']"),
    ("md", "## 4. Recovery of the known-rate gain by online estimation"),
    ("code", """rec = pd.read_csv('results/analysis/E1_recovery.csv')
print(rec[rec.oracle_gain_significant][['cfg_rho', 'cfg_coupling', 'online', 'oracle_gain', 'online_gain', 'recovery', 'rec_ci_lo', 'rec_ci_hi']].round(3).to_string(index=False))"""),
    ("md", "## 5. Monitoring history and non-stationarity (E2)"),
    ("code", """h = pd.read_csv('results/analysis/E2_gm_ratio_vs_EFT.csv'); h['pct'] = (h.gm_ratio - 1) * 100
print(h.pivot_table(index=['cfg_history_k', 'cfg_regime_shift', 'cfg_rho'], columns='policy', values='pct').round(2).to_string())
FIGS = ['figures/fig_history.png']"""),
    ("md", "## 6. Sensitivity (E3) and correlated shock (E4)"),
    ("code", """s = pd.read_csv('results/analysis/E3_sensitivity.csv'); s['pct'] = (s.gm_ratio - 1) * 100
print(s.pivot_table(index='variant', columns='policy', values='pct').round(2).to_string())
k = pd.read_csv('results/analysis/E4_shock.csv'); k['pct'] = (k.gm_ratio - 1) * 100
print(k.pivot_table(index='cfg_rho', columns='policy', values='pct').round(2).to_string())"""),
    ("md", "## 7. Tail behaviour and overhead"),
    ("code", """ov = pd.read_csv('results/analysis/E1_overhead_and_loss.csv')
print(ov.round(2).to_string(index=False))
FIGS = ['figures/fig_ecdf.png']"""),
    ("md", "## 8. v5.1 — change-aware history (E5) and bursty failures (E6)\nNew method: a per-VM likelihood-ratio test discards stale failure history. Held-out test seeds."),
    ("code", """subprocess.run([sys.executable, 'scripts/make_tables_v51.py'], check=True, cwd=ROOT)
subprocess.run([sys.executable, 'scripts/analyze_v51.py'], check=True, cwd=ROOT)
reg = pd.read_csv('results/analysis/E5_regret.csv', index_col=0)
print('E5 regret (percentage points):'); print(reg.round(2).to_string())
r6 = pd.read_csv('results/analysis/E6_regret.csv', index_col=0)
print('\\nE6 regret incl. Weibull failures:'); print(r6.round(2).to_string())
FIGS = ['figures/fig_e5_change_detection.png', 'figures/fig_regret.png', 'figures/fig_e6_bursty.png']"""),
    ("md", """## 9. Limitations
Exponential up-times, fixed repair, restart-from-scratch (no checkpointing), no communication
costs, no replication, one workflow at a time, 8 heterogeneous VMs, synthetic failure
processes on real workflow structures (no real failure traces). See the manuscript's
Threats to Validity section."""),
]


def main():
    os.chdir(ROOT)
    ns = {}
    cells = []
    for kind, src in CELLS:
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
            continue
        buf = io.StringIO()
        outputs = []
        t0 = time.time()
        ns["FIGS"] = []
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(src, "<cell>", "exec"), ns)
        except Exception:
            outputs.append({"output_type": "error", "ename": "Error", "evalue": "see traceback",
                            "traceback": traceback.format_exc().splitlines()})
            print(traceback.format_exc(), file=sys.stderr)
        text = buf.getvalue()
        if text:
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": text})
        for f in ns.get("FIGS", []):
            with open(os.path.join(ROOT, f), "rb") as fh:
                outputs.append({"output_type": "display_data", "metadata": {},
                                "data": {"image/png": base64.b64encode(fh.read()).decode(), "text/plain": f}})
        cells.append({"cell_type": "code", "execution_count": len([c for c in cells if c["cell_type"] == "code"]) + 1,
                      "metadata": {"wall_time_s": round(time.time() - t0, 2)}, "outputs": outputs, "source": src})
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": sys.version.split()[0]},
                                       "executed_by": "scripts/build_notebook.py"},
          "nbformat": 4, "nbformat_minor": 5}
    with open(OUT, "w") as fh:
        json.dump(nb, fh, indent=1)
    errs = sum(1 for c in cells for o in c.get("outputs", []) if o["output_type"] == "error")
    print(f"wrote {OUT}; cells={len(cells)}; errors={errs}")
    return errs


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
