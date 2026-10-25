# RESEARCH_PROGRESS

## Current direction (as of Stage 3)
Option **C** (narrowed question, AR-HEFT core replaced). RQ: when per-VM failure rates must
be estimated online, how much of the known-rate benefit of failure-aware placement is
recoverable, and when is failure-aware placement worth using? Reference rule: weight-free
expected-completion-time (ECT) placement. Details: `docs/RESEARCH_DESIGN.md`.

## Work completed
* Stage 1 audit: `docs/STAGE1_AUDIT.md` (code, math, data, provenance, literature).
* Stage 2–3 decision and pre-registered design: `docs/RESEARCH_DESIGN.md`.

## Files created or modified
* `v4_original/` — untouched copies of the supplied files, read-only, `SHA256SUMS`.
* `audit/v4_extracted.py` — v4 code cells extracted for re-execution.
* `docs/STAGE1_AUDIT.md`, `docs/RESEARCH_DESIGN.md`, this file.
* External data: WfCommons/WfInstances cloned (sparse, `pegasus/`) at commit
  `249866063b3bd7d6a36a5a55d4202c1f683099c0` into `/home/claude/ext/WfInstances` (not
  part of the package; selected instances are copied into `data/`).

## Commands executed (Stage 1)
* Re-analysis of `ar_heft_v4_100seed_raw.csv` (pandas pivot by seed).
* Reproduction attempt: 20 sampled scheduler runs re-executed with v4 code → 9 match.
* v4 probes: failure-free AR-HEFT vs EFT (90 fork-join instances); stuck-queue 3-task case.

## Results actually obtained (v4, supplied data)
* HEFT-ECT ≡ AR-HEFT-NoRisk in 2,700/2,700 paired runs.
* AR-HEFT mean makespan above HEFT-ECT in 8/9 cells (+0.19% to +1.06%).
* Utilisation × 8 = 0.998 for chain and "random" → no parallelism.

## Verified literature (title/venue checked against publisher, arXiv, or index pages)
* Yang, Zeng, Yang. TFR-GNN. arXiv:2609.09165 (2026).
* Wang, Ren, Yao, …, Zhu. Reliable cloud workflow scheduling with uncertain task execution time. JKSU-CIS 38:216 (2026). doi:10.1007/s44443-026-00585-x
* Oikonomou, Kolomvatsos, Anagnostopoulos. A proactive and uncertainty driven management mechanism for workflows of processing tasks. Computing 107:112 (2025). doi:10.1007/s00607-025-01468-1
* Mushtaq, Sheikh, Nain, Bharany, Ghoniem, Taye. CRFTS. Sci. Rep. 15:32233 (2025). doi:10.1038/s41598-025-17609-7
* "A robust fault-tolerant framework for VM failure predication [sic] and efficient task scheduling in dynamic cloud environments." JNCA 244:104340 (2025) — title/venue confirmed via ScienceDirect listing; **authors not yet verified** (page blocked).
* Aupy, Benoit, Casanova, Robert. Checkpointing strategies for scheduling computational workflows. IJNC 6(1):2–26 (2016). doi:10.15803/ijnc.6.1_2
* Topcuoglu, Hariri, Wu. HEFT. IEEE TPDS 13(3):260–274 (2002). doi:10.1109/71.993206
* Coleman et al. WfCommons. FGCS 128:16–27 (2022). doi:10.1016/j.future.2021.09.043
* Doğan, Özgüner. Trading off execution time for reliability… IPDPS 2001. doi:10.1109/IPDPS.2001.925005
* Doğan, Özgüner. Biobjective scheduling algorithms for execution time–reliability trade-off… Comput. J. 48(3):300–314 (2005). doi:10.1093/comjnl/bxh086

## Decisions and why
* Reject A: mechanism loses to its baseline and is uncalibrated. Reject D: would compete
  head-on with TFR-GNN without the means to train/evaluate learned policies here.
* Deadline handling dropped from the method (inert in v4); kept as a metric.
* WfCommons real instances replace synthetic DAGs (fixes degenerate "random" DAGs).

## Unresolved problems
* JNCA 2025 author list unverified.
* pip/PyPI unreachable from the shell; only stdlib + numpy/pandas/scipy/matplotlib/networkx.


---------------------------------------------------------------------------
## Update after Stages 4–5 (2026-10-24)

### Work completed
* `fawsim/` package (workflows, environment, estimators, policies, simulator, experiments).
* 23 unit tests (analytical Monte Carlo vs closed form, failure-free bounds, queue-release
  regression, restart semantics, H0 reduction to EFT on all 14 workflows, information-access
  denial, future-trace invariance, history visibility, pairing/determinism, estimators).
  Two mutation checks confirmed tests fail when the queue-release fix or outage counting is
  removed. Log: `results/test_logs/unittest_v5.txt`.
* Tuning passes on seeds 0–19 (`results/tuning*`): β=0.5 chosen for RT; pre-registered ECT
  rule found to lose to EFT at high exposure → deviation D1 recorded before test runs.
* Test-seed experiments E1–E4 (seeds 1000–1099) completed; raw CSV SHA-256 in
  `results/RAW_SHA256SUMS`; analysis in `results/analysis/`, figures in `figures/`.

### Commands executed
```
python -m unittest -v tests.test_fawsim
python scripts/run_experiments.py tuning | tuning_modes | tuning_rt_online
python scripts/run_experiments.py main | history | sensitivity | shock
python scripts/analyze.py && python scripts/make_tables.py
```
(E1 took 333 s on 2 cores; E2 157 s; E3 95 s; E4 27 s.)

### Results actually obtained (test seeds; geometric-mean makespan vs EFT)
* Known-rate gains exist essentially only when the volatile VMs are the fast ones:
  ECT −3.3% (ρ=0.03), −6.0% (ρ=0.1), −0.0% (ρ=0.3); RT −3.3%, −5.7%, −1.8%.
* Pre-registered ECT loses to EFT at ρ=0.3: uniform +8.1%, random +3.7%, slow +3.1%;
  max loss: +8.1% known rates, +9.0% online. RT's worst cell is +0.9%.
* Online estimation (no history) recovers 0.56–0.66 (ECT) and 0.41–0.56 (RT) of the
  known-rate gain in fast-volatile cells at ρ ∈ {0.03, 0.1}.
* Per-VM rates are the source of the gain: pooled-rate variants gain at most 2.0% (ECT) / 0.1% (RT).
* Stationary history raises RT recovery to 0.73 (k=5) and 0.89 (k=50) at ρ=0.1; under a
  regime shift stale history is harmful (RT +0.8 to +1.4% vs EFT) and forgetting (half-life
  H/4) only partly offsets it.
* Spare capacity and short repair magnify gains (m=16: −25% known, −20%/−14% online).
* AR-v4 legacy rule: higher mean makespan than EFT in 16/16 E1 cells (CI excludes 0 in 14; up to +9.6%).
* Overhead: 1.8 µs/decision (EFT) to 9.7 µs (ECT-Online).

### Decisions
* D1 (tuning): RT family added as labelled secondary; primary family unchanged.
* Wilcoxon (location) and the bootstrap/mean disagree in several cells because gains are
  in the tails; an exploratory sign-flip test of the mean is reported alongside.

### Unresolved
* No IEEEtran.cls offline → manuscript uses an IEEE-like article layout.
* JNCA 2025 author list still unverified (cited by title/venue only).

### Next concrete action
Stage 6: manuscript (LaTeX), bibliography, README, executed notebook.

---------------------------------------------------------------------------
## Update after Stages 6–7 (2026-10-24) — FINAL STATE / HANDOFF

### Work completed
* Manuscript `manuscript/main.tex` → `main.pdf` (7 pp., IEEE-like two-column `article`;
  swap to IEEEtran for submission). Tables generated from CSVs by `scripts/make_tables.py`;
  `manuscript/numbers.json` holds the quoted numbers. Verified bibliography `references.bib`.
* Executed notebook `FAWS_v5_Research_Notebook.ipynb` (18 cells, 0 errors) built by
  `scripts/build_notebook.py`.
* `README.md` (reproduction commands, verification scope, limitations), `CHANGELOG.md`.
* `choose_beta` scripted (reproduces beta = 0.5).

### Final checks executed
* 23/23 tests pass (re-run after all changes).
* `sha256sum -c results/RAW_SHA256SUMS`: 7/7 OK.
* 30 random E1 replicates (360 runs) re-executed: 0 mismatches vs `results/main/raw.csv`.
* 18 numerical claims in the manuscript checked programmatically against analysis CSVs: all
  match. Five overstatements found in the first draft were corrected before the check
  (pooled variants, ECTR, AR-v4 count, history ranges, prior sensitivity).

### Verdict
AR-HEFT v4: not publishable; retired. The v5 study supports a modest, honest paper: a
characterisation of when failure-aware placement pays off, a negative result for a
calibrated expected-time rule, and measured recovery under online estimation. Venue level:
a workshop or a short/regular paper in a systems venue that values careful empirical
studies; not a "new scheduler" paper.

### Unresolved / next actions for a future session
1. Verify JNCA 2025 (doi:10.1016/j.jnca.2025.104340) authors; check whether Girault et al.
   has a journal version to cite instead of the HAL report.
2. Port to IEEEtran; add author list and affiliations.
3. Strongest next experiment: real failure/revocation traces (e.g. spot interruption or
   failure-trace archives) in place of exponential up-times.
4. Optional: evaluate TFR-GNN-style learned placement given *estimated* rates, if code is
   released.

---------------------------------------------------------------------------
## Update v5.1 (2026-10-25) — novelty: change-aware history

### Direction
User asked for genuine novelty. Built on our own E2 finding (stale history hurts) and on
Failure Trace Archive evidence (real up-times Weibull, k ≈ 0.33–0.85).

### Work and results (test seeds unless stated)
* RT-HistCD (per-VM LR test, α = 0.05 chosen on tuning seeds). E5: beats plain history in
  12/12 shifted cells (10 Holm-sig), costs ≤ 0.29 points when stationary (none sig); best
  average non-oracle rule (−1.17% vs EFT). Worst-case regret 0.73 vs 2.35–3.73 (post hoc).
* E6 bursty failures: still beats history after shifts (10/24 sig, 0 worse); worst regret 0.76
  vs ≥ 2.92. Dispersion correction (HCDQ) not better → rejected.
* Age-aware Weibull hazard pilot (tuning seeds only): ≤ 0.13 points → abandoned, no claim.
* Literature: FTA (Kondo et al. CCGrid 2010; Javadi et al. JPDC 73(8) 2013) verified; SpotLake
  arXiv:2202.02973. No prior workflow-scheduling work found that tests history validity
  before use (searches recorded in conversation; absence of evidence, stated carefully).
* Verified: 31/31 tests; 180 sampled E5/E6 runs reproduce exactly; 21 new manuscript numbers
  checked against CSVs.

### Next
Presentation deck for professors; then real spot/failure traces (strongest next experiment).
