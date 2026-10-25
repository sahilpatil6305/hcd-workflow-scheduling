# Stage 2–3 — Research decision and design (pre-registered before any test-seed run)

Frozen: 2026-10-24, before running any test seed. Changes after this point are logged in
`CHANGELOG.md` with a reason; anything decided after seeing test-seed results is labelled
*post hoc* in the manuscript.

## 1. Decision

| Option | Novelty | Feasibility | Scientific value | Testability | Cost | Verdict |
|---|---|---|---|---|---|---|
| A. Keep AR-HEFT, fix bugs | None (components all known) | High | Low: it loses to its baseline | Yes | Low | Rejected — the mechanism, not just the code, is the problem |
| B. Redesign AR-HEFT's score, same question | Low: "reliability-aware HEFT with adaptive rates" is crowded | High | Medium | Yes | Medium | Partly adopted (the score is replaced) |
| **C. Narrow the question** | Modest but real: estimation error is unanalysed in the closest work | High | Medium–high: tells practitioners *when* failure-awareness pays and *how much data* it needs | Yes, with clean ablations | Medium | **Chosen** |
| D. New direction (e.g., learned GNN, replication) | Competing head-on with TFR-GNN / replication literature | Low here (no GPU, no training data) | Uncertain | Weak | High | Rejected |

**Selected direction (C with a B-style core replacement).** Replace AR-HEFT with a
weight-free *expected-completion-time* (ECT) placement rule, and study, on real WfCommons
Pegasus workflows:

> **RQ.** When per-VM failure rates are unknown and must be estimated from monitoring,
> how much of the makespan benefit of failure-aware placement (with known rates) can an
> online scheduler recover, and in which exposure and heterogeneity regimes is
> failure-aware placement worth using at all?

The contribution is a characterisation, a calibrated reference rule, and an honest
measurement — not a claim of a new best scheduler.

## 2. Contribution boundary

* **TFR-GNN** (Yang, Zeng & Yang, arXiv:2609.09165): learned placement, expected makespan,
  failure rates *given* as features. We do not compete with it; we study the
  given-vs-estimated gap that it leaves open. We do not reproduce its numbers.
* **Aupy, Benoit, Casanova & Robert** (IJNC 6(1), 2016): closed-form expected time under
  exponential failures with restart; we *use* this result (no novelty claimed for it).
* **Reliability-tilted HEFT** (Doğan & Özgüner 2001/2005; R-HEFT in TFR-GNN): weighted
  finish-time/reliability scores. We include a tuned member of this family as a baseline.
* **uHEFT** (Oikonomou et al., *Computing* 2025): online EFT with checkpointing/migration
  on revocable VMs; we do not model checkpointing.
* **Wang et al.** (*JKSU-CIS* 2026), **CRFTS** (Sci. Rep. 2025), **JNCA 244:104340** (2025):
  replication / reservation / prediction frameworks with different objectives (cost,
  independent tasks). Positioned in related work only.

## 3. System model

* VMs `j = 1..m`, speeds `s_j`. Each VM alternates up periods ~ Exp(λ_j(t)) and down periods
  of fixed length `R`. `λ_j` is piecewise-constant (constant in stationary experiments).
* Workflow DAG `G = (V, E)`, task work `w_i` (WfCommons `runtimeInSeconds`, rescaled so the
  mean task takes 1 time unit on a mean-speed VM). Duration on VM `j`: `d_ij = w_i / s_j`.
  No communication cost (stated limitation).
* Execution: one task at a time per VM, FIFO queue per VM, non-preemptive. An outage
  start (i) kills the running task — all its progress is lost; it becomes ready again after
  restart overhead `o`; (ii) **releases every queued task back to the ready pool** (v4
  defect fixed); (iii) makes the VM unavailable for `R`.
* Same dispatch loop for every policy: at each event, ready tasks are taken in decreasing
  upward rank (mean speed, no communication), each assigned to an *up* VM by the
  policy's score. Policies differ **only** in the score and the rate information they see.

## 4. Information available at a decision at time t

| Information | EFT | ECT-Oracle / RT-Oracle | ECT-Online / ECT-Pooled / AR-v4 | ECT-History(k) |
|---|---|---|---|---|
| DAG, works, speeds, queue states, VM up/down now | ✓ | ✓ | ✓ | ✓ |
| Outage start/end events with time ≤ t during this run | – | – | ✓ | ✓ |
| Outage events in the pre-run window [−H, 0) | – | – | – | ✓ |
| True current rates λ_j(t) (not the trace) | – | ✓ | – | – |
| Future outage times | **never** | **never** | **never** | **never** |

Enforced structurally: schedulers receive a `View` object; the trace lives only inside the
simulator. A regression test perturbs the trace after time t* and checks that every
decision before t* is unchanged.

## 5. Placement rules

Let `A_j` = time VM `j` can start the task (now + estimated backlog).

* **EFT**: `S_j = A_j + d_ij`.
* **ECT** (weight-free). With `p = 1 − e^{−λ_j d_ij}`:
  `S_j = A_j + E[C_ij]`, `E[C_ij] = E[min(T, d_ij)] + p (o + C*_i) = p/λ_j + p (o + C*_i)`,
  where `C*_i = min_{k≠j, up} (e^{λ_k d_ik} − 1)(1/λ_k + o)` is the Aupy et al. expected
  time of retry-until-success on the best alternative VM (its backlog ignored).
  As `λ_j → 0`, `S_j → A_j + d_ij` (implemented exactly: `λ = 0 ⇒ S_j = EFT score`).
* **RT-EFT** (reliability-tilted, R-HEFT-like): `S_j = A_j + d_ij + β d_ij (1 − e^{−λ_j d_ij})`,
  β chosen on tuning seeds from {0.5, 1, 2, 4, 8}, oracle rates.
* **AR-v4** (legacy): v4's risk score ported verbatim (weight 8, `d/2 + o`, adaptive
  multiplier, Gamma posterior); its deadline component is omitted because it was inert.

Rate estimators (for ECT-Online/History/Pooled and AR-v4):
* Gamma–Poisson posterior mean `λ̂_j = (a0 + n_j)/(b0 + U_j)` with `n_j` observed outage
  starts and `U_j` observed up-time. Prior mean `λ0 = a0/b0` = true pool-average rate
  (the *scale* is assumed known, e.g. from a provider's interruption statistics; the
  *per-VM* rates are what must be learned). `a0 = 1` fixed a priori. Misspecified prior
  (`λ0` × 0.25 and × 4) is a sensitivity experiment.
* **Pooled**: one pooled posterior for all VMs (isolates per-VM discrimination from mere
  exposure-awareness).
* **Discounted**: counts and up-time decayed with half-life `H/4` (fixed a priori).

## 6. Hypotheses (falsifiable; primary = H1, H2)

* **H0 (reduction).** With λ ≡ 0, every ECT variant produces a schedule identical to EFT.
* **H1 (value with known rates).** ECT-Oracle reduces geometric-mean makespan vs EFT.
  The reduction increases with exposure ρ; it is < 1% at ρ = 0.01; it is largest when
  volatile VMs are the fast ones and smallest when they are the slow ones.
* **H2 (cost of estimation).** ECT-Online (no history) recovers less than half of
  ECT-Oracle's gain (recovery fraction < 0.5) where the oracle gain is significant;
  recovery increases with history depth k ∈ {0, 5, 50}.
* **H3 (calibration).** ECT-Oracle is at least as good as RT-EFT-Oracle with its
  best tuned β; AR-v4 is worse than EFT at ρ ≤ 0.03 (replicating the v4 finding).
* **H4 (per-VM discrimination).** ECT-Oracle beats an exposure-only variant using the
  pooled oracle rate when rates are heterogeneous; they tie under uniform rates.
* **H5 (non-stationarity, secondary).** After a regime change at t = 0, discounted history
  beats undiscounted history at k = 50.

Each mechanism has an ablation: per-VM rates (H4: Pooled), estimation (H2: Oracle vs
Online vs History), calibration (H3: RT, AR-v4), forgetting (H5: Discounted).

## 7. Experimental factors

* Workloads: 14 WfCommons Pegasus instances (2 each of 1000genome, cycles, epigenomics,
  montage, seismology, soykb, srasearch; 42–201 tasks), repository commit recorded.
* Pool: m = 8, speeds as v4 `[1.00, 1.20, 0.82, 1.45, 0.95, 1.12, 0.72, 1.30]`.
* Exposure ρ = λ̄ · d̄ ∈ {0.01, 0.03, 0.1, 0.3} (λ̄ pool-mean rate, d̄ = 1).
* Heterogeneity / coupling: `uniform`; 2 of 8 VMs volatile with 10× the stable rate,
  chosen as the two fastest (`fast-volatile`), two slowest (`slow-volatile`) or at random
  per seed (`random-volatile`). Pool mean rate held at λ̄ in all cases.
* Repair `R = 1`, restart overhead `o = 0.05` (time units of d̄); sensitivity R ∈ {0.25, 4}.
* Seeds: tuning 0–19 (used only to pick β and check pipelines); test 1000–1099.

### 7a. Experiment list (added during Stage 4, before any test-seed run)

* **E0** validation tests (`tests/`).
* **T** tuning (seeds 0–19): choose RT β; pipeline check. Not used for any claim.
* **E1** main grid: 4 ρ × 4 couplings × 14 instances × 100 test seeds; policies EFT,
  ECT-Oracle, ECT-OraclePooled, ECT-Online, ECT-OnlinePooled, AR-v4, RT-Oracle(β*).
* **E2** history: random-volatile, ρ ∈ {0.1, 0.3}, k ∈ {5, 50}, stationary vs regime
  shift; EFT, ECT-Oracle, ECT-Online, ECT-Hist, ECT-HistDisc.
* **E3** sensitivity at ρ = 0.1, fast-volatile: R ∈ {0.25, 4}, volatile multiplier
  ∈ {3, 30}, m = 16 (speeds duplicated, 4 volatile), prior mean × {0.25, 4}.
* **E4** correlated shock (robustness, secondary): uniform rates ρ ∈ {0.03, 0.1} plus 3
  random VMs down for 5 time units at 0.3·M0. Question: does online learning from a
  one-off correlated event mislead the estimator (failure case)?
* Overhead: scheduling time per decision recorded in every run.

## 8. Metrics and analysis plan

* **Primary metric:** makespan ratio vs EFT on the same instance and trace,
  `r = M_policy / M_EFT`; summarised as geometric mean (mean of log r).
* Inference: per (ρ, coupling) cell, paired over (instance, seed); 95% percentile
  bootstrap CI of mean log r (10,000 resamples, resampling seeds within instance);
  two-sided Wilcoxon signed-rank; Holm correction over the primary family
  {ECT-Oracle, ECT-Online} × 4 ρ × 4 couplings (32 tests).
* **Recovery fraction** `(1 − GM_online)/(1 − GM_oracle)` with bootstrap CI, reported only
  where the oracle CI excludes 0.
* Secondary: deadline-meeting probability `P(M ≤ δ · M0)` for δ ∈ {1.1, 1.25, 1.5}, M0 =
  failure-free EFT makespan (a metric only; no policy uses deadlines); lost work;
  scheduling time per decision; per-application breakdown.
* Effect-size floor: we call a difference "practically negligible" if |GM − 1| < 1%.

## 9. Threats and alternative explanations

* *Gains from avoiding slow VMs, not failures* → ruled out by H0 (identical when λ = 0).
* *Gains from the fallback term rather than rate knowledge* → Pooled ablation (H4).
* *One application drives the average* → per-application table.
* *Oracle advantage partly from regime knowledge* → Oracle only knows rates, not times.
* *Simulator artefact* → analytical Monte Carlo tests of the restart formula and outage
  process; trace-invariance test; failure-free equality test.
* *Model realism*: exponential up-times, fixed repair, no checkpointing, no communication,
  no replication, single workflow at a time. Stated as limitations; not tested.

## 10. Deviations from the frozen design (all decided on tuning seeds 0–19, before any test-seed run)

**D1 — the pre-registered ECT rule under-performed on tuning seeds; a secondary rule family
was added.** On tuning seeds, ECT-Oracle (fallback mode, the pre-registered rule) was
worse than EFT by 7.5% (geometric mean) at ρ = 0.3 with uniform rates and by 2.1% with
random-volatile rates, with more lost work (210 vs 152 work units) — it sends long tasks
to slow VMs because its fallback term assumes an immediate, backlog-free retry elsewhere.
Two alternative ECT modes (backlog-aware fallback; Aupy et al. retry-in-place) were tried on
tuning seeds; neither fixed it (worst-cell losses 7.9% and 4.5%). A reliability-tilted EFT
with a small weight (RT, β = 0.5) was the best rule overall on tuning seeds (−0.8% mean
log-ratio; worst cell +0.8%).

Consequences, fixed before test runs:
* The pre-registered primary family is **unchanged** (ECT-Oracle and ECT-Online, fallback
  mode, vs EFT; 32 Holm-corrected tests). Its likely negative outcome will be reported.
* A **secondary family** is added and labelled as added after tuning: RT with β = 0.5 for
  every rate source (Oracle, OraclePooled, Online, OnlinePooled, Hist, HistDisc), so
  oracle-vs-online comparisons differ only in information. (On tuning seeds RT-Online's
  own best β was 0.25, mean −0.33% vs −0.26% at 0.5 — immaterial; we keep 0.5.)
* ECTR (retry-in-place) Oracle/Online is reported descriptively only.
* H2 (recovery fraction) and H5 (forgetting) are evaluated for both families.
* H3 is evaluated as registered (ECT-Oracle vs RT-Oracle); tuning already suggests it
  will be refuted, and it is reported either way.

## 11. Addendum v5.1 — stale-history detection (frozen 2026-10-25, before any E5 run)

**Motivation (from E2, test seeds).** Monitoring history raised RT's recovery of the
known-rate gain to 0.73–0.89 when conditions were stationary, but after a regime change it
made RT 0.8–1.4% *worse* than EFT; exponential forgetting only partly helped.

**Method (RT-HistCD).** Same RT score (β = 0.5). Rate source: per VM, keep history
(t < 0) and in-run (t ≥ 0) counts separately; compute the likelihood-ratio statistic for
equality of two Poisson rates; if it exceeds the χ²(1) critical value for level α, use
prior + in-run data only for that VM, else pool history and in-run data. No other change.

**Hypotheses.**
* H6a (stationary): RT-HistCD is within ±0.5 percentage points of RT-Hist.
* H6b (regime shift): RT-HistCD is better than RT-Hist and not worse than RT-Online by more
  than 0.5 points.
* H6c (overall): averaged over stationary and shifted conditions, RT-HistCD has the lowest
  geometric-mean makespan among RT-Online, RT-Hist, RT-HistDisc, RT-HistCD.

**Tuning.** α ∈ {0.001, 0.01, 0.05}, chosen on seeds 0–19 by the lowest mean log ratio to EFT
over the whole E5 grid. Default if tuning is flat: α = 0.01.

**E5 grid (test seeds 1000–1099).** coupling ∈ {random-volatile, fast-volatile} ×
ρ ∈ {0.03, 0.1, 0.3} × k ∈ {5, 50} × {stationary, shift} × 14 workflows. Policies: EFT,
RT-Oracle, RT-Online, RT-Hist, RT-HistDisc, RT-HistCD(α*), and ECT-Hist / ECT-HistCD
(descriptive). Analysis: as in §8 (paired GM ratio, stratified bootstrap CIs); pairwise
RT-HistCD vs RT-Hist and vs RT-Online with Holm correction over the 24 cells × 2 contrasts
(sign-flip test of the mean, since §10 showed the mean is the relevant estimand).

## 12. Addendum v5.1b — robustness to bursty (non-exponential) failures (frozen 2026-10-25, before any E6 run)

**Motivation.** Kondo et al. (Failure Trace Archive, CCGrid 2010; JPDC journal version)
report Weibull shape parameters of 0.33–0.85 for real availability intervals, i.e.
long-tailed, bursty failures. The LR test in §11 assumes Poisson counts; overdispersion
should cause false "stale" alarms and discard useful history.

**Method (RT-HistCDQ).** Same as RT-HistCD (α = 0.05 from §11 tuning; no new tuning), but
the statistic is compared with crit × φ, φ = max(1, mean over VMs of the squared CV of
completed history up-times, VMs with ≥ 3 intervals). For a renewal process, count variance
≈ CV² × mean (Cox), so φ ≈ 1 for exponential up-times.

**Hypotheses.**
* H7a: under Weibull up-times (shape 0.5, 0.7) and stationarity, RT-HistCD loses part of
  RT-Hist's benefit (false alarms); RT-HistCDQ recovers most of it.
* H7b: under a regime shift, RT-HistCDQ remains better than RT-Hist.
* H7c: with exponential up-times, RT-HistCDQ ≈ RT-HistCD (|Δ| < 0.3 points).

**E6 grid (test seeds 1000–1099).** up-time shape ∈ {1.0, 0.7, 0.5} × coupling ∈
{random-volatile, fast-volatile} × ρ ∈ {0.03, 0.1} × k ∈ {5, 50} × {stationary, shift} ×
14 workflows. Policies: EFT, RT-Oracle, RT-Online, RT-Hist, RT-HistDisc, RT-HistCD,
RT-HistCDQ (all β = 0.5, α = 0.05). Mean of each up-time distribution equals 1/λ.

## 13. Outcomes of §11–§12 and an abandoned pilot (2026-10-25)

* E5 (test seeds): H6a supported (stationary cost ≤ 0.29 points, none significant);
  H6b first part supported (better than RT-Hist in 12/12 shifted cells, 10/12 Holm-significant),
  second part violated in 3 of 12 cells (up to +0.73 points vs RT-Online); H6c supported.
* E6 (test seeds): H7a weak — false-alarm cost of RT-HistCD under Weibull up-times is small
  (≤ 0.79 points; 1/16 cells significant) and RT-HistCDQ recovers only part of it; H7b
  supported for k = 0.7, weak for k = 0.5; H7c supported (|Δ| ≤ 0.15). Net: the dispersion
  correction is not worth adding; RT-HistCD remains the method.
* Analysis change: Holm correction for E5/E6 contrasts uses paired t-test p-values because the
  sign-flip p-value (B = 2,000) cannot fall below 5e-4, which made Holm over 192 E6 contrasts
  unable to reject anything. Sign-flip p-values are kept in the CSVs.
* **Abandoned (tuning seeds only, no test run):** an age-aware Weibull-hazard RT
  (RT-HistCDA). Change vs RT-HistCD: −0.04 (k=0.5), −0.06 (k=0.7), +0.13 (exponential)
  points — below the 1% practical-relevance floor. Not evaluated on test seeds; no claim made.
