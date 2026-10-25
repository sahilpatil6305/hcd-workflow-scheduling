# CHANGELOG

## v5.1.0 — 2026-10-25 (novel contribution: change-aware history)

* **New estimator `ChangeAwareEstimator`** (`fawsim/estimators.py`): per-VM likelihood-ratio
  test for two Poisson rates between pre-run history and in-run observations; stale history
  is discarded per VM. Policy names `RT-HistCD-b<β>-a<α>`, `ECT-HistCD-a<α>`.
* Variant with quasi-Poisson dispersion correction (`RT-HistCDQ…`) — evaluated, not recommended.
* Exploratory `AgeAwareRT` (`RT-HistCDA…`, Weibull hazard + VM age) — piloted on tuning seeds
  only, abandoned (effect < 0.15 points).
* **Weibull up-times** (`EnvConfig.uptime_shape`); seeds of all v5.0 experiments unchanged
  (regression test `test_v50_seeds_unchanged`).
* Experiments: `tuning_cd` (α = 0.05), `e5` (stale history), `tuning_age` (pilot), `e6` (bursty).
* Analysis: `scripts/analyze_v51.py`, `scripts/make_tables_v51.py`; Holm on paired t-tests for
  E5/E6 contrasts (sign-flip resolution too coarse for 192 tests; documented in design §13).
* Manuscript and Overleaf project updated (`scripts/make_overleaf.py`). Tests: 23 → 31.

## v5.0.0 — 2026-10-24 (this package; supersedes AR-HEFT v4)

### Research direction
* AR-HEFT is retired as a method (see `docs/STAGE1_AUDIT.md`). v5 studies when
  failure-aware placement helps and how much of the known-rate benefit survives online
  estimation (`docs/RESEARCH_DESIGN.md`).

### Simulator defects fixed (v4 → v5)
1. **Queued tasks no longer stay bound to a failed VM.** On an outage start, every task
   queued on that VM is released to the ready pool (regression test
   `test_queue_release_regression`: 3-task case 51.0 → 21.5).
2. **Degenerate "random" DAGs removed.** v4's generator always included the chain i→i+1, so
   "random" DAGs were total orders. Workloads are now 14 WfCommons Pegasus instances
   (regression test `test_instances_have_parallelism`).
3. **Prior no longer leaks the true rate per VM.** v4 passed the scenario's true base rate
   as the prior mean (and never used `PRIOR_RATE`). v5 sets the prior to the pool-average
   rate (documented assumption) and tests misspecification (×0.25, ×4).
4. **Failure trace is exact and extendable.** Per-VM independent RNG streams
   (`SeedSequence.spawn`); horizon extension preserves the prefix; runs that pass the
   horizon are re-run with a longer one (never silently failure-free).
5. **Shock no longer targets a fixed VM subset that serial work avoids.** Shocks hit a
   random subset per seed (E4).
6. **Information access is enforced**, not just documented: schedulers receive a `View`;
   oracle rates raise `InformationViolation` for non-oracle policies; a test changes the
   future trace and checks decisions before that time are identical.
7. **Documentation matches code**: restart semantics (all progress lost) are stated once.
8. Deadlines are a *metric* (P(M ≤ δ·M0)), not a scheduler input; v4's deadline component
   was bit-identical to its baseline in all 2,700 v4 runs.

### Methods
* Removed: AR-HEFT scoring (kept only as the labelled legacy port `AR-v4`).
* Added: `EFT`, `ECT` (modes fallback / fallback_backlog / inplace), `RT` (reliability-tilted
  EFT), rate sources oracle / oracle_pooled / online / online_pooled / history /
  history_disc, Gamma–Poisson estimator with optional forgetting.

### Process
* Tuning seeds 0–19 and test seeds 1000–1099 are disjoint. Deviation D1 (tuning-based
  addition of the RT family) is recorded in `docs/RESEARCH_DESIGN.md` §10 before any
  test-seed run.
* v4 files preserved read-only under `v4_original/` with SHA-256 sums.
