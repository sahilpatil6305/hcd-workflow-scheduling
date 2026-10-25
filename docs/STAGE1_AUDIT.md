# Stage 1 — Audit of AR-HEFT v4 and its premise

Date: 2026-10-24. All numbers below were computed in this session from the supplied files
(`v4_original/`, SHA-256 in `v4_original/SHA256SUMS`) using `audit/v4_extracted.py`
(the notebook's code cells, unmodified except `display` → `print`).

## Verdict (short)

**AR-HEFT v4 should not be submitted, and its core mechanism should not be kept.**
It is a combination of known components (online HEFT/EFT dispatch, a reliability-tilted
placement score, a Gamma–Poisson rate estimate, slack-based ordering); its risk penalty is
not a calibrated quantity; its deadline mechanism does nothing measurable; and on the
supplied 100-seed data it is *worse* than its own failure-oblivious baseline. The simulator
has defects that make two thirds of the experiment uninformative. One part of the idea
survives scrutiny: the question of what happens when a reliability-aware scheduler must
*estimate* failure rates online is open in the closest recent work (TFR-GNN assumes the
rates are given and explicitly does not analyse estimation error).

## 1. Is the problem important and specific?

Reliability-aware placement for workflows on failure-prone resources is important (spot /
preemptible VMs, volunteer and edge resources), but the v4 question bundles adaptive
hazard learning, deadline feasibility, and correlated shocks without isolating any single
claim. No hypothesis in v4 is stated in a falsifiable form.

## 2. Contribution beyond known heuristics

| AR-HEFT component | Closest prior work (verified) |
|---|---|
| Online upward-rank EFT dispatch with failure events | uHEFT in Oikonomou et al., *Computing* 107:112 (2025) |
| Finish time + weighted failure-probability term | Reliability/makespan trade-off of Doğan & Özgüner (IPDPS 2001; *Comput. J.* 48(3) 2005); "R-HEFT" reliability-tilted HEFT used as a baseline in TFR-GNN (arXiv:2609.09165) |
| Failure probability `1-exp(-λ·d)` | Standard (e.g., Girault, Saule & Trystram, "Reliability versus performance for critical applications") |
| Expected cost of restart after failure | Exact closed form already known: `E[t] = (1/λ + D)(e^{λw} − 1)` (Aupy, Benoit, Casanova & Robert, *IJNC* 6(1):2–26, 2016) |
| Online estimation of revocation likelihood | Oikonomou et al. 2025 estimate revocation probability online (kernel density over recent demand) |
| Deadline slack ordering | Partial-critical-path sub-deadlines, e.g. Wang et al., *JKSU-CIS* 38:216 (2026) |

No component is new, and v4 presents no evidence that the combination produces an effect.

## 3. Mathematical coherence

1. **Risk penalty is not a calibrated expectation.** v4 adds `8 · p_fail · (0.5·d + 0.25)`.
   The expected time lost to a failure that occurs before completion is
   `E[T | T < d] = 1/λ − d·e^{−λd}/(1−e^{−λd})`, not `d/2`, and the subsequent
   re-execution cost is omitted entirely. The factor `RISK_WEIGHT = 8` then rescales a time
   quantity by an arbitrary constant, so the score is no longer an expected finish time.
2. **Adaptive multiplier double-counts.** When outages rise, the per-VM posterior already
   rises; v4 additionally multiplies the weight by up to 2.5× using the pooled observed rate.
3. **Slack guard is ad hoc.** It scales the risk weight by remaining slack with a floor of
   0.10; there is no objective this approximates.
4. **Prior leaks the truth.** `simulate_workflow` passes the scenario's true `base_rate` as the
   prior mean (`PRIOR_RATE` is defined but never used). In the stable mode the prior *is*
   the true rate.
5. **Penalty distorts failure-free schedules.** With the nominal prior (0.012) and *no
   outages*, AR-HEFT-NoDeadline is slower than EFT on 60% of 90 fork-join instances (mean
   +0.7% at n=12, +1.1% at n=24). The v4 "reduces to baseline" test only passes because it
   sets the prior to zero.

## 4. Can the simulator produce valid evidence?

| Defect | Evidence | Consequence |
|---|---|---|
| "random" DAGs are total orders | Generator always adds every edge i→i+1; checked: all present. Mean utilisation × 8 VMs = 0.998 for both `chain` and `random` | 2 of 3 workload families have zero parallelism; "random" duplicates "chain" |
| Queued tasks stay bound to a VM that goes down | 3-task case: makespan 51.0 instead of ≈21 | Penalises whichever scheduler queues behind a soon-to-fail VM; not a scheduler property |
| Shock never hits the VM serial workloads use | Shock VMs are 0,1,2; EFT runs serial work on VM 3 (fastest) | "shock" condition is not stress for chain/random |
| Exposure regime is tiny | λ·d ≈ 0.066 for an average task on the fastest VM | Ceiling on any reliability-aware gain is small; differences are dominated by noise |
| Deadline = 1.05 × the baseline's own failure-free makespan | — | Violation is mostly "did any failure hit the critical path" |
| **Supplied CSV does not reproduce from the supplied notebook** | 9 of 20 sampled runs match; fork-join failure-free makespans differ (28.11 vs 28.19) | The results were produced by a different code revision; provenance is broken |
| Docs say partial progress is retained; code discards it | Cell text vs `rt.remaining = tasks[...].work` | Documentation inconsistency |

## 5. Are the baselines fair and representative?

* `STATIC-RA-Oracle` is not an oracle (it knows nominal rates, not the trace). With uniform
  rates it is bit-identical to HEFT-ECT in every stable and shock run.
* `AR-HEFT-NoRisk` is **bit-identical to HEFT-ECT in all 2,700 paired runs**, so the deadline
  mechanism has no measurable effect and the "ablation" is the baseline.
* No published method is implemented.

## 6. What do the supplied experiments support?

AR-HEFT vs HEFT-ECT, 100 paired seeds per cell (from the supplied raw CSV):

* Mean makespan is higher for AR-HEFT in 8 of 9 structure × mode cells (+0.19% to +1.06%);
  the one exception is fork-join/shock (−0.01%).
* AR-HEFT is never better in more than 8.3% of pairs in any cell and is worse in 10–43%.
* Deadline-violation rate: chain 0.366 vs 0.302; random 0.381 vs 0.308; fork-join 0.773 vs
  0.772.

**Supported conclusion:** under these conditions AR-HEFT is slightly but consistently
worse than failure-oblivious EFT, its deadline component is inert, and its risk component
is harmful. This is a negative result about this heuristic, not evidence for it.

## 7. Would a different formulation be stronger?

The closest recent competitor, TFR-GNN (Yang, Zeng & Yang, arXiv:2609.09165), already
minimises expected makespan under per-machine Poisson failures with restart, uses WfCommons
Pegasus workflows, matches HEFT exactly when failure-free, and compares against a
reliability-tilted HEFT. It **takes the failure rates as inputs** and does not analyse
what happens when they are estimated. That gap is narrow but real, testable in this
simulator, and is the only part of AR-HEFT's motivation ("adaptive") that survives.
