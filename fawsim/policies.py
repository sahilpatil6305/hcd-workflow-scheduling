"""Placement policies. All share the simulator's dispatch loop (upward-rank order);
they differ only in the VM score and in the information they are allowed to see.

Information flags (enforced by the simulator):
  uses_monitor : receives outage start/end events with time <= now during the run
  uses_history : additionally receives the pre-run monitoring window [-H, 0)
  uses_oracle  : may query the true current hazard regime (never outage times)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from .estimators import ChangeAwareEstimator, RateEstimator


# ---------------------------------------------------------------------------
# Closed-form building blocks
# ---------------------------------------------------------------------------
def expected_attempt_time(lam: float, d: float) -> float:
    """E[min(T, d)] for T ~ Exp(lam): (1 - e^{-lam d}) / lam; equals d at lam = 0."""
    if lam <= 0.0:
        return d
    return -math.expm1(-lam * d) / lam


def p_fail(lam: float, d: float) -> float:
    return 0.0 if lam <= 0.0 else -math.expm1(-lam * d)


def expected_retry_time(lam: float, d: float, downtime: float) -> float:
    """Aupy et al. (2016): expected time to complete d units of work by retrying from
    scratch after each failure with `downtime` lost per failure: (1/lam + D)(e^{lam d} - 1).
    Equals d at lam = 0. Returns inf on overflow."""
    if lam <= 0.0:
        return d
    x = lam * d
    if x > 700:
        return math.inf
    return (1.0 / lam + downtime) * math.expm1(x)


def ect_cost(lam_j: float, d_j: float, overhead: float, fallback: float) -> float:
    """Expected completion time on VM j when a failure triggers re-execution elsewhere:
    E[min(T,d)] + p (o + C*). Equals d_j exactly when lam_j = 0."""
    if lam_j <= 0.0:
        return d_j
    p = -math.expm1(-lam_j * d_j)
    return p / lam_j + p * (overhead + fallback)


# ---------------------------------------------------------------------------
class Policy:
    name = "base"
    uses_monitor = False
    uses_history = False
    uses_oracle = False

    def reset(self, ctx: "SimContext") -> None:
        self.ctx = ctx

    # monitoring hooks (only called if uses_monitor)
    def begin_observation(self, t: float, up_state: List[bool]) -> None:
        pass

    def observe_start(self, j: int, t: float) -> None:
        pass

    def observe_end(self, j: int, t: float) -> None:
        pass

    def choose(self, view, tid: int, work: float, cands: Sequence[int], A: Dict[int, float]) -> int:
        raise NotImplementedError


class SimContext:
    def __init__(self, speeds, overhead, prior_mean, history_start, repair=0.0):
        self.speeds = list(speeds)
        self.overhead = overhead
        self.prior_mean = prior_mean
        self.history_start = history_start
        self.repair = repair          # platform parameter (known), not trace information


def _argmin(scores):
    # deterministic tie-break on VM id
    return min(scores, key=lambda x: (x[0], x[1]))[1]


# ---------------------------------------------------------------------------
class EFT(Policy):
    name = "EFT"

    def choose(self, view, tid, work, cands, A):
        sp = self.ctx.speeds
        return _argmin([(A[j] + work / sp[j], j) for j in cands])


class _RateSource:
    """Mixin providing rates() from oracle or an estimator."""
    source = "oracle"   # oracle | oracle_pooled | online | online_pooled | history | history_disc | history_cd
    a0 = 1.0
    prior_scale = 1.0
    cd_crit = 6.635     # chi-square(1) critical value for history_cd (alpha = 0.01)

    def _init_source(self):
        s = self.source
        self.uses_oracle = s.startswith("oracle")
        self.uses_monitor = not self.uses_oracle
        self.uses_history = s.startswith("history")

    def reset(self, ctx):
        super().reset(ctx)
        self.est = None
        if self.uses_monitor:
            hl = None
            if self.source == "history_disc":
                H = -ctx.history_start
                hl = H / 4.0 if H > 0 else None
            if self.source in ("history_cd", "history_cdq"):
                self.est = ChangeAwareEstimator(len(ctx.speeds), ctx.prior_mean * self.prior_scale,
                                                a0=self.a0, crit=self.cd_crit,
                                                quasi=self.source == "history_cdq")
            else:
                self.est = RateEstimator(len(ctx.speeds), ctx.prior_mean * self.prior_scale,
                                         a0=self.a0, half_life=hl,
                                         pooled=self.source.endswith("pooled"))

    def begin_observation(self, t, up_state):
        if self.est is not None:
            self.est.start(t, up_state)

    def observe_start(self, j, t):
        self.est.outage_start(j, t)

    def observe_end(self, j, t):
        self.est.outage_end(j, t)

    def rates(self, view) -> List[float]:
        if self.uses_oracle:
            r = view.oracle_rates()
            if self.source == "oracle_pooled":
                mean = sum(r) / len(r)
                return [mean] * len(r)
            return r
        return self.est.rates(view.now)


class ECT(_RateSource, Policy):
    """Expected-completion-time placement (no tunable weight).

    mode = "fallback"         : pre-registered rule. E[min(T,d)] + p (o + C*), C* = Aupy retry
                                cost on the best *other* VM, its backlog ignored.
    mode = "fallback_backlog" : as above but C* includes the alternative VM's backlog.
    mode = "inplace"          : Aupy et al. retry-in-place cost on VM j itself,
                                (1/lam_j + max(R, o)) (e^{lam_j d} - 1).
    All three return exactly the EFT score when lam_j = 0.
    """

    def __init__(self, source="oracle", name=None, prior_scale=1.0, a0=1.0, mode="fallback"):
        self.source = source
        self.prior_scale = prior_scale
        self.a0 = a0
        self.mode = mode
        self._init_source()
        self.name = name or f"ECT-{mode}-{source}"

    def choose(self, view, tid, work, cands, A):
        sp, o = self.ctx.speeds, self.ctx.overhead
        lam = self.rates(view)
        if self.mode == "inplace":
            D = max(self.ctx.repair, o)
            return _argmin([(A[j] + expected_retry_time(lam[j], work / sp[j], D), j) for j in cands])
        m = len(sp)
        retry = [expected_retry_time(lam[k], work / sp[k], o) for k in range(m)]
        if self.mode == "fallback_backlog":
            now = view.now
            alt = [(A[k] - now + retry[k]) if k in A else math.inf for k in range(m)]
        else:
            alt = retry
        order = sorted(range(m), key=lambda k: (alt[k], k))
        best, second = order[0], (order[1] if m > 1 else order[0])
        scores = []
        for j in cands:
            d = work / sp[j]
            fb = alt[second] if j == best else alt[best]
            scores.append((A[j] + ect_cost(lam[j], d, o, fb), j))
        return _argmin(scores)


class RTEFT(_RateSource, Policy):
    """Reliability-tilted EFT (R-HEFT family): A + d + beta * d * p_fail."""

    def __init__(self, beta=2.0, source="oracle", name=None):
        self.beta = beta
        self.source = source
        self._init_source()
        self.name = name or f"RT-{source}-b{beta:g}"

    def choose(self, view, tid, work, cands, A):
        sp = self.ctx.speeds
        lam = self.rates(view)
        return _argmin([(A[j] + work / sp[j] + self.beta * (work / sp[j]) * p_fail(lam[j], work / sp[j]), j)
                        for j in cands])


class ARv4(_RateSource, Policy):
    """Legacy port of AR-HEFT v4's risk score (AR-HEFT-NoDeadline placement rule).

    hazard: Gamma posterior with prior exposure 4 d_bar (v4: 30 time units with d_bar ~7.5);
    score = finish + 8 * mult * p * (0.5 d + o); mult = min(2.5, 1 + 0.35 max(0, obs/prior - 1)).
    The v4 deadline component is omitted: it was bit-identical to the baseline in v4 data.
    """
    name = "AR-v4"
    RISK_WEIGHT, KAPPA, CAP, PRIOR_EXPOSURE = 8.0, 0.35, 2.5, 4.0

    def __init__(self):
        self.source = "online"
        self._init_source()

    def reset(self, ctx):
        Policy.reset(self, ctx)
        lam0 = ctx.prior_mean
        a0 = max(lam0 * self.PRIOR_EXPOSURE, 1e-12)
        self.est = RateEstimator(len(ctx.speeds), lam0, a0=a0) if lam0 > 0 else None

    def begin_observation(self, t, up_state):
        if self.est:
            self.est.start(t, up_state)

    def observe_start(self, j, t):
        if self.est:
            self.est.outage_start(j, t)

    def observe_end(self, j, t):
        if self.est:
            self.est.outage_end(j, t)

    def choose(self, view, tid, work, cands, A):
        sp, o = self.ctx.speeds, self.ctx.overhead
        if self.est is None:
            return _argmin([(A[j] + work / sp[j], j) for j in cands])
        lam = self.est.rates(view.now)
        N = U = 0.0
        for j in range(len(sp)):
            n, u = self.est._counts(j, view.now)
            N += n
            U += u
        observed = N / max(U, 1e-9)
        mult = min(self.CAP, 1.0 + self.KAPPA * max(0.0, observed / self.ctx.prior_mean - 1.0))
        scores = []
        for j in cands:
            d = work / sp[j]
            pen = p_fail(lam[j], d) * (0.5 * d + o)
            scores.append((A[j] + d + self.RISK_WEIGHT * mult * pen, j))
        return _argmin(scores)


def _chi2_crit(alpha: float) -> float:
    from scipy.stats import chi2
    return float(chi2.ppf(1.0 - alpha, 1))


PRIMARY_ECT_MODE = "fallback"   # pre-registered; see CHANGELOG for any change


class AgeAwareRT(RTEFT):
    """v5.2 (exploratory): RT whose failure probability uses a Weibull hazard and each VM's
    observed age (time since it last came back up).

    shape k: estimated once at t=0 from the CV^2 of completed history up-times (pooled over
    VMs); per-VM mean up-time 1/lambda_j from the change-aware estimator. For a task that
    would start on VM j at A_j, age a = A_j - (last time j came up, or start of observation):
        p_j = 1 - exp(-((a+d)/s_j)^k + (a/s_j)^k),  s_j = (1/lambda_j) / Gamma(1+1/k).
    With k = 1 this is exactly 1 - exp(-lambda_j d), i.e. RT-HistCD.
    """

    def __init__(self, beta=0.5, name=None, crit=None, fixed_shape=None):
        super().__init__(beta=beta, source="history_cd", name=name or f"RT-HistCDA-b{beta:g}")
        if crit is not None:
            self.cd_crit = crit
        self.fixed_shape = fixed_shape
        self.k = None

    def _shape(self):
        if self.k is None:
            from .estimators import pooled_cv2, weibull_shape_from_cv2
            if self.fixed_shape is not None:
                self.k = self.fixed_shape
            else:
                cv2 = pooled_cv2(self.est._ups)
                self.k = 1.0 if cv2 is None else weibull_shape_from_cv2(cv2)
        return self.k

    def choose(self, view, tid, work, cands, A):
        sp = self.ctx.speeds
        lam = self.rates(view)
        k = self._shape()
        g = math.gamma(1.0 + 1.0 / k)
        scores = []
        for j in cands:
            d = work / sp[j]
            if lam[j] <= 0:
                p = 0.0
            else:
                s_j = (1.0 / lam[j]) / g
                since = self.est.last_up_since[j]
                if since is None:
                    since = self.est.obs_start if self.est.obs_start is not None else 0.0
                a = max(0.0, A[j] - since)
                x0, x1 = (a / s_j) ** k, ((a + d) / s_j) ** k
                p = -math.expm1(-(x1 - x0))
            scores.append((A[j] + d + self.beta * d * p, j))
        return _argmin(scores)


def make_policy(name: str) -> Policy:
    """Registry used by experiment configs."""
    if name == "EFT":
        return EFT()
    if name == "AR-v4":
        return ARv4()
    if name.startswith("RT-HistCDA"):
        # RT-HistCDA-b<beta>[-a<alpha>][-k<fixed shape>]
        body = name[len("RT-HistCDA"):]
        alpha = shape = None
        if "-k" in body:
            body, kk = body.rsplit("-k", 1); shape = float(kk)
        if "-a" in body:
            body, a = body.rsplit("-a", 1); alpha = float(a)
        beta = float(body.split("-b", 1)[1])
        return AgeAwareRT(beta=beta, name=name, crit=_chi2_crit(alpha) if alpha else None, fixed_shape=shape)
    if name.startswith("RT-"):
        # RT-<Source>-b<beta>, Source in Oracle/OraclePooled/Online/OnlinePooled/Hist/HistDisc
        body, scale = name[3:], 1.0
        if "-prior" in body:
            body, sc = body.split("-prior")
            scale = float(sc)
        alpha = None
        if "-a" in body:
            body, a = body.rsplit("-a", 1)
            alpha = float(a)
        src, beta = body.rsplit("-b", 1)
        rt_sources = {"Oracle": "oracle", "OraclePooled": "oracle_pooled", "Online": "online",
                      "OnlinePooled": "online_pooled", "Hist": "history", "HistDisc": "history_disc",
                      "HistCD": "history_cd", "HistCDQ": "history_cdq"}
        if src not in rt_sources:
            raise ValueError(f"unknown policy {name}")
        pol = RTEFT(beta=float(beta), source=rt_sources[src], name=name)
        pol.prior_scale = scale
        if alpha is not None:
            pol.cd_crit = _chi2_crit(alpha)
        return pol
    # ECT names: "ECT-<Source>" uses the primary mode (PRIMARY_ECT_MODE);
    # "ECTF-/ECTB-/ECTR-<Source>" force fallback / fallback_backlog / inplace.
    sources = {"Oracle": "oracle", "OraclePooled": "oracle_pooled", "Online": "online",
               "OnlinePooled": "online_pooled", "Hist": "history", "HistDisc": "history_disc",
               "HistCD": "history_cd", "HistCDQ": "history_cdq"}
    modes = {"ECT": PRIMARY_ECT_MODE, "ECTF": "fallback", "ECTB": "fallback_backlog", "ECTR": "inplace"}
    parts = name.split("-")
    if parts[0] in modes and len(parts) >= 2 and parts[1] in sources:
        scale, alpha = 1.0, None
        for extra in parts[2:]:
            if extra.startswith("prior"):
                scale = float(extra[5:])
            elif extra.startswith("a"):
                alpha = float(extra[1:])
            else:
                raise ValueError(f"unknown policy {name}")
        pol = ECT(source=sources[parts[1]], name=name, prior_scale=scale, mode=modes[parts[0]])
        if alpha is not None:
            pol.cd_crit = _chi2_crit(alpha)
        return pol
    raise ValueError(f"unknown policy {name}")
