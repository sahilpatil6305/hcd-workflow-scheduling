# HCD: Change-Aware Failure Estimation for Reliability-Aware Workflow Scheduling

Code, data and results for the paper **“Change-Aware Failure Estimation for Reliability-Aware
Workflow Scheduling”** (Sahil Patil, IIT (BHU) Varanasi; draft, October 2026).

**HCD** (history with change detection) is a failure-rate estimator for online workflow
schedulers. For each VM it runs a likelihood-ratio test comparing the failure rate in the
monitoring history with the rate observed in the current run, and discards that VM's history
when the two disagree. Stale failure history otherwise makes failure-aware placement worse than
ignoring failures. The estimator lives in `fawsim/estimators.py` (`ChangeAwareEstimator`);
the scheduler that uses it is `RT-HistCD-b0.5-a0.05` in `fawsim/policies.py`.

### Quick start

```bash
git clone https://github.com/sahilpatil6305/hcd-workflow-scheduling.git
cd hcd-workflow-scheduling
python -m unittest -v tests.test_fawsim          # 31 tests, ~30 s
gunzip -k results/*/raw.csv.gz                   # raw results (compressed for GitHub's size limit)
python scripts/make_tables_v51.py && python scripts/analyze_v51.py   # reproduce E5/E6 tables and figures
```

Requirements: Python ≥ 3.10, numpy, pandas, scipy, matplotlib (tested with Python 3.13.17,
numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, matplotlib 3.11.2). After `gunzip`,
`sha256sum -c results/RAW_SHA256SUMS` verifies the raw results.

---

# fawsim v5 — failure-aware workflow placement under estimated failure rates

Successor package to **AR-HEFT v4**. After an audit (`docs/STAGE1_AUDIT.md`) AR-HEFT was
retired: it was slightly worse than its own baseline, its deadline component was inert,
and its simulator had defects. The project was narrowed to one question:

> When per-VM failure rates must be estimated online, how much of the makespan benefit of
> failure-aware placement (with known rates) can be recovered, and when is failure-aware
> placement worth using at all?

## Bottom line (test seeds, 14 WfCommons workflows × 100 seeds per cell)

* Known-rate failure-aware placement helps materially **only when the unreliable VMs are
  the fast ones**: −3.3% (ρ=0.03) and −6.0% (ρ=0.1) geometric-mean makespan vs EFT.
  Elsewhere the attainable gain is < 1%.
* The pre-registered weight-free expected-completion-time rule (**ECT**) is **worse than
  EFT** at high exposure (up to +8.1% with known rates). A lightly reliability-tilted EFT
  (**RT**, β = 0.5) never loses more than 0.9%.
* Online estimation without history recovers 41–66% of the known-rate gain; a stationary
  monitoring history 73–89%; stale history after a regime change is harmful.
* **New (v5.1): change-aware history.** A per-VM likelihood-ratio test that discards stale
  failure history is never more than 0.73 points from the best rule across 24 held-out
  conditions (alternatives: 2.35–3.73), and 0.76 vs ≥ 2.92 including bursty Weibull failures.
* See `manuscript/main.pdf` (or `overleaf/` for the IEEE version).

## Layout

| Path | Contents |
|---|---|
| `fawsim/` | simulator (`simulator.py`), environments/traces (`environment.py`), estimators, policies, WfCommons loader, experiment runner |
| `tests/test_fawsim.py` | 23 analytical, invariant, information-access and regression tests |
| `scripts/run_experiments.py` | all experiments (tuning, E1 main, E2 history, E3 sensitivity, E4 shock) |
| `scripts/analyze.py` | pre-registered analysis → `results/analysis/*.csv`, `figures/` |
| `scripts/make_tables.py` | LaTeX tables + `manuscript/numbers.json` from the analysis CSVs |
| `scripts/build_notebook.py` | builds and executes `FAWS_v5_Research_Notebook.ipynb` (no Jupyter needed) |
| `data/wfcommons/` | 14 extracted Pegasus instances + `MANIFEST.json` (source URLs at a pinned commit, SHA-256) |
| `results/<exp>/raw.csv.gz`, `config.json` | every run, one row per (workflow, environment, seed, policy); `gunzip -k` before analysis |
| `results/RAW_SHA256SUMS` | checksums of all raw CSVs |
| `results/test_logs/unittest_v5.txt` | test output from the run that preceded the experiments |
| `docs/` | Stage 1 audit, pre-registered design with deviation log (§10) |
| `manuscript/` | `main.tex`, `main.pdf`, `references.bib`, generated `tables/` |
| `overleaf/` | IEEEtran version of the paper, ready to upload to Overleaf |
| `v4_original/` | the supplied v4 notebook and CSVs, read-only, with `SHA256SUMS` |
| `audit/v4_extracted.py` | v4 code cells extracted for re-execution in the audit |
| `CHANGELOG.md`, `RESEARCH_PROGRESS.md` | changes and running research log |

## Reproduce

Tested with Python 3.13.17, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, matplotlib 3.11.2
(networkx is not required). Times are on 2 cores.

```bash
python -m unittest -v tests.test_fawsim           # ~30 s, 31 tests
python scripts/run_experiments.py tuning          # seeds 0-19, ~60 s
python scripts/run_experiments.py tuning_modes    # ~50 s
python scripts/run_experiments.py tuning_rt_online
python scripts/run_experiments.py choose_beta     # writes results/tuning/chosen_beta.json (beta = 0.5)
python scripts/run_experiments.py main            # E1, ~330 s, 268,800 runs
python scripts/run_experiments.py history         # E2, ~160 s
python scripts/run_experiments.py sensitivity     # E3, ~95 s
python scripts/run_experiments.py shock           # E4, ~30 s
python scripts/analyze.py                         # tables + figures
python scripts/make_tables.py                     # manuscript tables
python scripts/build_notebook.py                  # executed notebook
# v5.1
python scripts/run_experiments.py tuning_cd       # chooses alpha on seeds 0-19 (~70 s)
python scripts/run_experiments.py e5              # stale-history experiment (~500 s)
python scripts/run_experiments.py e6              # bursty failures (~730 s)
python scripts/run_experiments.py tuning_age      # abandoned pilot (tuning seeds only)
python scripts/make_tables_v51.py && python scripts/analyze_v51.py
python scripts/make_overleaf.py                   # IEEEtran project + overleaf_paper.zip
cd manuscript && latexmk -pdf main.tex
```

Seeds: tuning 0–19, test 1000–1099. Per-replicate traces are derived from
`SeedSequence([seed, crc32(instance), crc32(environment)])` and are identical across
policies and across machines; `sha256sum -c results/RAW_SHA256SUMS` should pass after a
rerun on the same library versions.

## What was and was not verified

* **Executed here:** all tests, all experiments, all analysis, the notebook (by
  `build_notebook.py`, which runs every code cell and stores the real outputs), and the
  LaTeX build.
* **Not available here:** Jupyter itself (the notebook was not run in a Jupyter kernel),
  `IEEEtran.cls` (the draft uses an IEEE-like `article` layout; switch the documentclass
  for submission), PyPI (WfCommons instances were taken directly from GitHub).
* **Literature:** titles, venues and DOIs were checked against publisher/arXiv/index pages.
  The author list of the 2025 JNCA paper (doi:10.1016/j.jnca.2025.104340) could not be
  retrieved and is marked as unverified.

## Limitations

Synthetic failure processes (exponential up-times, fixed repair) on real workflow
structures and runtimes; restart from scratch, no checkpointing, no replication, no
communication costs, one workflow at a time; one 8-VM pool (plus a 16-VM variant); one tilt
weight. TFR-GNN, uHEFT and replication-based methods are discussed, not reimplemented.
The RT family was added after a tuning pass (documented before any test run) and is
reported as secondary. Wilcoxon (pre-registered) and the mean-based estimand disagree in
several cells; both are reported.
