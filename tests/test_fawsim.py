"""Analytical, invariant and regression tests for fawsim v5.  Run: python -m unittest -v"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fawsim.environment import (EnvConfig, Environment, V5_SPEEDS, make_environment, merge,
                                no_failure_environment, rates_for, volatile_set)
from fawsim.estimators import RateEstimator
from fawsim.experiments import INSTANCES, MEAN_SPEED, get_workflow, run_replicate
from fawsim.policies import (ECT, EFT, ARv4, ect_cost, expected_attempt_time, expected_retry_time,
                             make_policy)
from fawsim.simulator import InformationViolation, View, simulate
from fawsim.workflows import Task, Workflow, critical_path_lb, from_edges, validate

ALL_POLICIES = ["EFT", "ECT-Oracle", "ECT-OraclePooled", "ECT-Online", "ECT-OnlinePooled",
                "ECT-Hist", "ECT-HistDisc", "RT-Oracle-b2", "AR-v4", "ECTF-Online", "ECTB-Online",
                "ECTR-Online", "ECTR-Hist", "RT-HistCD-b0.5", "ECT-HistCD", "RT-HistCDQ-b0.5-a0.05"]


def custom_env(trace, speeds, rates=None, repair=1.0, overhead=0.05, history_start=0.0):
    m = len(speeds)
    rates = rates or [0.0] * m
    cfg = EnvConfig(rho=sum(rates) / m, speeds=tuple(speeds), repair=repair, overhead=overhead)
    return Environment(cfg, list(rates), list(rates), (), (), history_start,
                       {j: merge(trace.get(j, [])) for j in range(m)}, horizon=1e12)


class TestWorkflows(unittest.TestCase):
    def test_instances_valid_and_normalised(self):
        for inst in INSTANCES:
            wf = get_workflow(inst)
            validate(wf)
            mean_dur = np.mean([t.work for t in wf.tasks]) / MEAN_SPEED
            self.assertAlmostEqual(mean_dur, 1.0, places=9, msg=inst)

    def test_cycle_rejected(self):
        with self.assertRaises(ValueError):
            from_edges("c", "x", [1, 1, 1], [(0, 1), (1, 2), (2, 0)])

    def test_instances_have_parallelism(self):
        # regression for the v4 defect (degenerate total-order "random" DAGs)
        from fawsim.workflows import max_width
        for inst in INSTANCES:
            self.assertGreater(max_width(get_workflow(inst)), 1, inst)


class TestEnvironment(unittest.TestCase):
    def test_rates_mean_and_coupling(self):
        rng = np.random.default_rng(0)
        for c in ("uniform", "fast-volatile", "random-volatile", "slow-volatile"):
            cfg = EnvConfig(rho=0.1, coupling=c)
            vol = volatile_set(cfg, rng)
            r = rates_for(cfg, vol)
            self.assertAlmostEqual(np.mean(r), 0.1, places=12)
            if c != "uniform":
                self.assertAlmostEqual(max(r) / min(r), 10.0, places=9)
        fast = volatile_set(EnvConfig(rho=0.1, coupling="fast-volatile"), rng)
        self.assertEqual(set(fast), {3, 7})   # speeds 1.45 and 1.30
        slow = volatile_set(EnvConfig(rho=0.1, coupling="slow-volatile"), rng)
        self.assertEqual(set(slow), {6, 2})   # speeds 0.72 and 0.82

    def test_deterministic_disjoint_and_prefix_stable(self):
        cfg = EnvConfig(rho=0.3, coupling="random-volatile", history_k=5)
        e1 = make_environment(cfg, [7, 8, 9], 200.0)
        e2 = make_environment(cfg, [7, 8, 9], 200.0)
        self.assertEqual(e1.trace, e2.trace)
        e3 = make_environment(cfg, [7, 8, 9], 800.0)       # longer horizon
        for j in range(cfg.m):
            ivs = e1.trace[j]
            for (a, b), (c, d) in zip(ivs, ivs[1:]):
                self.assertLess(b, c + 1e-12)
            self.assertEqual(ivs, [iv for iv in e3.trace[j] if iv[0] < 200.0])
        self.assertAlmostEqual(e1.history_start, -5 / 0.3)

    def test_uptime_distribution(self):
        # alternating renewal: mean up-time 1/lambda, fixed repair
        cfg = EnvConfig(rho=0.5, coupling="uniform", repair=2.0)
        env = make_environment(cfg, [1], 40000.0)
        ups = []
        for ivs in env.trace.values():
            prev_end = 0.0
            for a, b in ivs:
                self.assertAlmostEqual(b - a, 2.0)
                ups.append(a - prev_end)
                prev_end = b
        ups = np.array(ups[1:])
        self.assertLess(abs(ups.mean() - 2.0) / (ups.std() / math.sqrt(len(ups))), 4.0)

    def test_weibull_uptime_mean(self):
        cfg = EnvConfig(rho=0.5, coupling="uniform", repair=1.0, uptime_shape=0.5)
        env = make_environment(cfg, [2], 60000.0)
        ups = []
        for ivs in env.trace.values():
            prev = 0.0
            for a, b in ivs:
                ups.append(a - prev); prev = b
        ups = np.array(ups)
        self.assertLess(abs(ups.mean() - 2.0) / (ups.std() / math.sqrt(len(ups))), 4.0)
        self.assertGreater(ups.std() / ups.mean(), 1.5)      # Weibull k=0.5: CV = sqrt(5) ~ 2.24

    def test_v50_seeds_unchanged(self):
        # regression: adding uptime_shape must not change v5.0 traces / results
        import pandas as pd
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "main", "raw.csv")
        path = base if os.path.exists(base) else base + ".gz"
        if not os.path.exists(path):
            self.skipTest("results/main/raw.csv(.gz) not present")
        df = pd.read_csv(path, nrows=2000)
        row = df.iloc[1234]
        rows = run_replicate(row.instance, EnvConfig(rho=row.cfg_rho, coupling=row.cfg_coupling), int(row.seed), [row.policy])
        self.assertEqual(rows[0]["makespan"], row.makespan)

    def test_regime_shift_disjoint(self):
        cfg = EnvConfig(rho=0.1, coupling="random-volatile", history_k=50, regime_shift=True)
        env = make_environment(cfg, [3], 100.0)
        self.assertTrue(set(env.run_volatile).isdisjoint(env.hist_volatile))
        self.assertNotEqual(env.true_rates(-1.0), env.true_rates(1.0))


class TestClosedForms(unittest.TestCase):
    def test_attempt_time_monte_carlo(self):
        rng = np.random.default_rng(1)
        lam, d = 0.7, 1.9
        T = rng.exponential(1 / lam, 400000)
        self.assertAlmostEqual(np.minimum(T, d).mean(), expected_attempt_time(lam, d), delta=0.005)

    def test_limits(self):
        self.assertEqual(expected_attempt_time(0.0, 2.5), 2.5)
        self.assertEqual(expected_retry_time(0.0, 2.5, 1.0), 2.5)
        self.assertEqual(ect_cost(0.0, 2.5, 0.1, 99.0), 2.5)
        self.assertAlmostEqual(expected_retry_time(1e-9, 2.5, 1.0), 2.5, places=6)

    def test_single_vm_restart_matches_aupy_formula(self):
        """Simulator vs closed form: one task, one VM, restart after repair R >= o:
        E[C] = (1/lam + R)(e^{lam d} - 1)."""
        lam, d, R = 0.4, 2.0, 0.6
        wf = Workflow("one", "t", (Task(0, d, (), ()),))
        cfg = EnvConfig(rho=lam, speeds=(1.0,), repair=R, overhead=0.05)
        ms = []
        for s in range(3000):
            env = make_environment(cfg, [s, 99], 1e4)
            ms.append(simulate(wf, env, EFT(), prior_mean=lam)["makespan"])
        ms = np.array(ms)
        expect = expected_retry_time(lam, d, R)
        se = ms.std() / math.sqrt(len(ms))
        self.assertLess(abs(ms.mean() - expect), 4 * se, f"{ms.mean():.3f} vs {expect:.3f} (se {se:.3f})")


class TestSimulator(unittest.TestCase):
    def test_failure_free_bounds(self):
        env = no_failure_environment(V5_SPEEDS)
        for inst in INSTANCES:
            wf = get_workflow(inst)
            r = simulate(wf, env, EFT(), prior_mean=0.0)
            self.assertEqual(r["work_conserved"], 1)
            self.assertEqual(r["lost_work"], 0.0)
            self.assertGreaterEqual(r["makespan"] + 1e-9, critical_path_lb(wf, max(V5_SPEEDS)))
            self.assertGreaterEqual(r["makespan"] + 1e-9, wf.total_work() / sum(V5_SPEEDS))
            self.assertLessEqual(r["utilization"], 1 + 1e-9)

    def test_queue_release_regression(self):
        """v4 left queued tasks on a failed VM until repair (makespan 51). v5 releases them."""
        wf = from_edges("q", "t", [10.0, 10.0, 10.0], [(0, 1), (0, 2)])
        env = custom_env({0: [(1.5, 50.0)]}, [10.0, 1.0], overhead=0.05)
        r = simulate(wf, env, EFT(), prior_mean=0.0)
        self.assertAlmostEqual(r["makespan"], 21.5, places=9)
        self.assertEqual(r["interruptions"], 1)
        self.assertEqual(r["released_queued"], 1)
        self.assertAlmostEqual(r["lost_work"], 5.0, places=9)

    def test_restart_discards_progress(self):
        wf = from_edges("one", "t", [10.0], [])
        env = custom_env({0: [(4.0, 5.0)], 1: [(0.0, 100.0)]}, [1.0, 1.0], overhead=0.25)
        r = simulate(wf, env, EFT(), prior_mean=0.0)
        self.assertAlmostEqual(r["lost_work"], 4.0)
        self.assertAlmostEqual(r["makespan"], 15.0)      # restart at 5.0 when VM0 is back
        self.assertEqual(r["work_conserved"], 1)

    def test_reduction_to_eft_without_failures(self):
        """H0: with lambda == 0 every ECT variant makes EFT's decisions."""
        env = make_environment(EnvConfig(rho=0.0), [0], 1e6)
        for inst in INSTANCES:
            wf = get_workflow(inst)
            ref = simulate(wf, env, EFT(), prior_mean=0.0, record=True)
            for p in ["ECT-Oracle", "ECT-OraclePooled", "ECT-Online", "ECT-OnlinePooled", "AR-v4", "RT-Oracle-b2",
                      "ECTF-Oracle", "ECTB-Oracle", "ECTB-Online", "ECTR-Oracle", "ECTR-Online"]:
                r = simulate(wf, env, make_policy(p), prior_mean=0.0, record=True)
                self.assertEqual(r["decision_log"], ref["decision_log"], f"{p} {inst}")

    def test_oracle_access_denied_to_online_policies(self):
        wf = get_workflow(INSTANCES[0])
        env = make_environment(EnvConfig(rho=0.1, coupling="fast-volatile"), [0], 1e5)

        class Cheater(EFT):
            name = "cheater"

            def choose(self, view, tid, work, cands, A):
                view.oracle_rates()
                return super().choose(view, tid, work, cands, A)

        with self.assertRaises(InformationViolation):
            simulate(wf, env, Cheater())

    def test_future_trace_invariance(self):
        """Decisions before t* must not depend on outages after t*."""
        inst = INSTANCES[2]
        wf = get_workflow(inst)
        cfg = EnvConfig(rho=0.3, coupling="random-volatile", history_k=5)
        env = make_environment(cfg, [11], 1e5)
        for p in ALL_POLICIES:
            r1 = simulate(wf, env, make_policy(p), record=True)
            tstar = r1["makespan"] / 2
            new_trace = {}
            for j, ivs in env.trace.items():
                keep = [iv for iv in ivs if iv[0] < tstar]
                last_end = keep[-1][1] if keep else -1e9
                s = max(tstar + 0.01 * (j + 1), last_end + 0.01)
                new_trace[j] = keep + [(s, s + 3.0)]     # different future
            env2 = Environment(env.cfg, env.run_rates, env.hist_rates, env.run_volatile,
                               env.hist_volatile, env.history_start, new_trace, env.horizon)
            r2 = simulate(wf, env2, make_policy(p), record=True)
            d1 = [d for d in r1["decision_log"] if d[0] < tstar]
            d2 = [d for d in r2["decision_log"] if d[0] < tstar]
            self.assertTrue(len(d1) > 0)
            self.assertEqual(d1, d2, p)

    def test_history_only_seen_by_history_policies(self):
        # Online and History estimators diverge only because of pre-run events.
        cfg = EnvConfig(rho=0.3, coupling="fast-volatile", history_k=50)
        env = make_environment(cfg, [5], 1e5)
        wf = get_workflow(INSTANCES[6])
        a = make_policy("ECT-Online"); simulate(wf, env, a)
        b = make_policy("ECT-Hist"); simulate(wf, env, b)
        self.assertLess(sum(a.est.n), sum(b.est.n))

    def test_paired_trace_and_determinism(self):
        cfg = EnvConfig(rho=0.1, coupling="random-volatile")
        r1 = run_replicate(INSTANCES[4], cfg, 1234, ALL_POLICIES)
        r2 = run_replicate(INSTANCES[4], cfg, 1234, ALL_POLICIES)
        self.assertEqual([x["makespan"] for x in r1], [x["makespan"] for x in r2])
        self.assertEqual(len({x["n_outage_intervals_run"] for x in r1}), 1)
        self.assertEqual(len({x["run_volatile"] for x in r1}), 1)
        self.assertTrue(all(x["work_conserved"] == 1 for x in r1))


class TestPolicyRegistry(unittest.TestCase):
    def test_names(self):
        from fawsim import policies
        self.assertEqual(make_policy("ECTR-Online-prior4").prior_scale, 4.0)
        self.assertEqual(make_policy("ECTR-Online").mode, "inplace")
        self.assertEqual(make_policy("ECT-Oracle").mode, policies.PRIMARY_ECT_MODE)
        with self.assertRaises(ValueError):
            make_policy("ECT-Nope")

    def test_inplace_score_is_aupy_formula(self):
        from fawsim.policies import SimContext
        pol = make_policy("ECTR-Oracle")
        pol.reset(SimContext([1.0, 2.0], 0.05, 0.1, 0.0, repair=1.0))

        class V:
            now = 0.0
            def oracle_rates(self):
                return [0.5, 2.0]
        # VM0: (1/0.5 + 1)(e^{0.5*4}-1) = 19.17 ; VM1: (0.5+1)(e^{2*2}-1) = 80.4 -> VM0
        self.assertEqual(pol.choose(V(), 0, 4.0, [0, 1], {0: 0.0, 1: 0.0}), 0)


class TestChangeAware(unittest.TestCase):
    def _feed(self, est, hist, run):
        est.start(-100.0, [True, True])
        for j, t, kind in hist + run:
            (est.outage_start if kind == "s" else est.outage_end)(j, t)

    def test_lr_statistic(self):
        from fawsim.estimators import ChangeAwareEstimator as C
        self.assertEqual(C.lr_stat(5, 50, 1, 10), 0.0)        # identical rates
        self.assertGreater(C.lr_stat(20, 100, 0, 50), 6.635)  # 0.2 vs 0 over 50 units
        self.assertAlmostEqual(C.lr_stat(4, 20, 1, 20), 2 * (4 * math.log(4 / 2.5) + 1 * math.log(1 / 2.5)))

    def test_consistent_history_is_pooled(self):
        from fawsim.estimators import ChangeAwareEstimator
        a = ChangeAwareEstimator(2, 0.1, crit=6.635); b = RateEstimator(2, 0.1)
        ev = [(0, -80.0, "s"), (0, -79.0, "e"), (0, -30.0, "s"), (0, -29.0, "e"), (0, 20.0, "s"), (0, 21.0, "e")]
        self._feed(a, ev[:4], ev[4:]); self._feed(b, ev[:4], ev[4:])
        for x, y in zip(a.rates(40.0), b.rates(40.0)):
            self.assertAlmostEqual(x, y, places=12)
        self.assertFalse(any(a.stale_flags))

    def test_stale_history_dropped(self):
        from fawsim.estimators import ChangeAwareEstimator
        a = ChangeAwareEstimator(2, 0.1, crit=6.635)
        hist = []
        for k in range(20):                       # VM0 failed 20 times in history
            hist += [(0, -100.0 + 5 * k, "s"), (0, -99.5 + 5 * k, "e")]
        self._feed(a, hist, [])
        r = a.rates(60.0)                         # no outage in 60 units of run time
        self.assertTrue(a.stale_flags[0])
        self.assertAlmostEqual(r[0], 1.0 / (10.0 + 60.0))   # prior + run only
        self.assertFalse(a.stale_flags[1])

    def test_dispersion(self):
        from fawsim.estimators import ChangeAwareEstimator as C
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(C.dispersion([list(rng.exponential(1, 4000))]), 1.0, delta=0.12)
        w = list(rng.weibull(0.5, 20000))
        self.assertAlmostEqual(C.dispersion([w]), 5.0, delta=1.2)       # CV^2 = Gamma(5)/Gamma(3)^2 - 1 = 5
        self.assertEqual(C.dispersion([[1.0, 1.0]]), 1.0)               # too few intervals -> 1

    def test_quasi_equals_plain_for_exponential_up_to_phi(self):
        cfg = EnvConfig(rho=0.1, coupling="random-volatile", history_k=50, regime_shift=True)
        env = make_environment(cfg, [9], 1e5)
        wf = get_workflow(INSTANCES[5])
        q = make_policy("RT-HistCDQ-b0.5-a0.05"); simulate(wf, env, q)
        self.assertLess(q.est.phi, 1.6)      # exponential history -> phi close to 1

    def test_no_history_equals_online(self):
        cfg = EnvConfig(rho=0.1, coupling="fast-volatile")
        env = make_environment(cfg, [3], 1e5)
        wf = get_workflow(INSTANCES[3])
        r1 = simulate(wf, env, make_policy("RT-HistCD-b0.5"), record=True)
        r2 = simulate(wf, env, make_policy("RT-Online-b0.5"), record=True)
        self.assertEqual(r1["decision_log"], r2["decision_log"])


class TestEstimator(unittest.TestCase):
    def test_gamma_posterior_mean(self):
        est = RateEstimator(2, prior_mean=0.1, a0=1.0)
        est.start(0.0, [True, True])
        est.outage_start(0, 4.0); est.outage_end(0, 5.0); est.outage_start(0, 9.0)
        r = est.rates(12.0)
        self.assertAlmostEqual(r[0], (1 + 2) / (10 + 8.0))     # up 0-4 and 5-9
        self.assertAlmostEqual(r[1], (1 + 0) / (10 + 12.0))

    def test_discount_limits(self):
        a = RateEstimator(1, 0.1, half_life=1e9); b = RateEstimator(1, 0.1)
        for e in (a, b):
            e.start(0.0, [True]); e.outage_start(0, 3.0); e.outage_end(0, 4.0)
        self.assertAlmostEqual(a.rates(10.0)[0], b.rates(10.0)[0], places=6)
        c = RateEstimator(1, 0.1, half_life=1.0)
        c.start(0.0, [True]); c.outage_start(0, 1.0); c.outage_end(0, 1.5)
        # long after the event the estimate returns towards (a0 + ~0)/(b0 + tau)
        tau = 1.0 / math.log(2)
        self.assertAlmostEqual(c.rates(1000.0)[0], 1.0 / (10.0 + tau), places=6)

    def test_pooled(self):
        e = RateEstimator(3, 0.2, pooled=True)
        e.start(0.0, [True] * 3); e.outage_start(1, 2.0)
        r = e.rates(2.0)
        self.assertEqual(len(set(r)), 1)
        self.assertAlmostEqual(r[0], 2.0 / (5.0 + 6.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
