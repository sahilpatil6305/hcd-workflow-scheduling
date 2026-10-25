"""Deterministic event-driven simulator for one workflow on a failure-prone VM pool.

Semantics (v5):
* one task at a time per VM; per-VM FIFO queue; non-preemptive;
* outage start: the running task loses all progress and becomes ready again after the
  restart overhead o; every task queued on that VM is released to the ready pool at once;
  the VM is unavailable until the outage ends;
* same-time events are processed in the order finish < outage_end < outage_start < retry,
  after which idle VMs start their queue heads and the dispatcher assigns ready tasks;
* schedulers see a View: never the trace. Monitoring events are forwarded only to policies
  with `uses_monitor`; pre-run history only to `uses_history`; true rates only to
  `uses_oracle`.
"""
from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional

from .environment import Environment
from .policies import Policy, SimContext
from .workflows import Workflow, upward_ranks

FINISH, OUT_END, OUT_START, RETRY = 0, 1, 2, 3
EPS = 1e-9


class HorizonExceeded(RuntimeError):
    pass


class InformationViolation(RuntimeError):
    pass


class View:
    """What a policy may inspect at a decision point."""

    def __init__(self, sim: "_Sim"):
        self._sim = sim

    @property
    def now(self) -> float:
        return self._sim.now

    def oracle_rates(self) -> List[float]:
        if not self._sim.policy.uses_oracle:
            raise InformationViolation(f"{self._sim.policy.name} is not an oracle policy")
        return self._sim.env.true_rates(self._sim.now)


class _Sim:
    def __init__(self, wf: Workflow, env: Environment, policy: Policy, prior_mean: float,
                 record: bool):
        self.wf, self.env, self.policy = wf, env, policy
        self.speeds = list(env.cfg.speeds)
        self.m = len(self.speeds)
        self.o = env.cfg.overhead
        self.now = 0.0
        self.record = record
        self.decisions: List[tuple] = []
        policy.reset(SimContext(self.speeds, self.o, prior_mean, env.history_start, env.cfg.repair))


def simulate(wf: Workflow, env: Environment, policy: Policy, prior_mean: Optional[float] = None,
             record: bool = False, max_events: int = 20_000_000) -> Dict:
    if prior_mean is None:
        prior_mean = sum(env.run_rates) / len(env.run_rates)
    S = _Sim(wf, env, policy, prior_mean, record)
    view = View(S)
    m, speeds, o = S.m, S.speeds, S.o
    mean_speed = sum(speeds) / m
    ranks = upward_ranks(wf, mean_speed)
    n = wf.n
    work = [t.work for t in wf.tasks]

    status = ["blocked"] * n          # blocked | pending | queued | running | done
    left = [len(t.parents) for t in wf.tasks]
    retry_after = [0.0] * n
    start_t = [0.0] * n
    run_vm = [-1] * n
    token = [0] * n
    done_t = [math.nan] * n
    for i in range(n):
        if left[i] == 0:
            status[i] = "pending"
    pending = set(i for i in range(n) if status[i] == "pending")

    up = [True] * m
    running: List[Optional[int]] = [None] * m
    queue: List[List[int]] = [[] for _ in range(m)]

    heap: List[tuple] = []
    serial = 0

    def push(t, kind, j=-1, tid=-1, tok=-1):
        nonlocal serial
        serial += 1
        heapq.heappush(heap, (t, kind, serial, j, tid, tok))

    # ---- outage trace: split into history (<0) and run (>=0) --------------------
    hist_events = []
    for j, ivs in env.trace.items():
        for a, b in ivs:
            if a < 0:
                hist_events.append((a, OUT_START, j))
                if b <= 0:
                    hist_events.append((b, OUT_END, j))
                else:
                    up[j] = False          # down at time 0
                    push(b, OUT_END, j)
            else:
                push(a, OUT_START, j)
                push(b, OUT_END, j)
    if policy.uses_monitor:
        if policy.uses_history and hist_events:
            hist_events.sort()
            policy.begin_observation(env.history_start, [True] * m)
            for t, kind, j in hist_events:
                (policy.observe_start if kind == OUT_START else policy.observe_end)(j, t)
        elif policy.uses_history:
            policy.begin_observation(env.history_start, [True] * m)
        else:
            policy.begin_observation(0.0, list(up))

    useful = lost = 0.0
    interruptions = released = observed_outages = 0
    decisions = 0
    sched_time = 0.0
    events = 0
    busy = 0.0

    def start_idle():
        nonlocal busy
        for j in range(m):
            if up[j] and running[j] is None and queue[j]:
                tid = queue[j].pop(0)
                status[tid] = "running"
                run_vm[tid] = j
                start_t[tid] = S.now
                token[tid] += 1
                running[j] = tid
                push(S.now + work[tid] / speeds[j], FINISH, j, tid, token[tid])

    def backlog():
        A = {}
        for j in range(m):
            if not up[j]:
                continue
            a = S.now
            r = running[j]
            if r is not None:
                a = max(a, start_t[r] + work[r] / speeds[j])
            for q in queue[j]:
                a += work[q] / speeds[j]
            A[j] = a
        return A

    def dispatch():
        nonlocal decisions, sched_time
        ready = [i for i in pending if retry_after[i] <= S.now + EPS]
        if not ready:
            return
        cands = [j for j in range(m) if up[j]]
        if not cands:
            return
        ready.sort(key=lambda i: (-ranks[i], i))
        A = backlog()
        for tid in ready:
            t0 = time.perf_counter()
            j = policy.choose(view, tid, work[tid], cands, A)
            sched_time += time.perf_counter() - t0
            decisions += 1
            if j not in A:
                raise RuntimeError(f"{policy.name} chose unavailable VM {j}")
            pending.discard(tid)
            status[tid] = "queued"
            run_vm[tid] = j
            queue[j].append(tid)
            A[j] += work[tid] / speeds[j]
            if S.record:
                S.decisions.append((S.now, tid, j))
        start_idle()

    start_idle()
    dispatch()
    remaining = n
    while remaining:
        if not heap:
            raise RuntimeError("deadlock: no events but unfinished tasks")
        t = heap[0][0]
        if t > env.horizon:
            raise HorizonExceeded(f"simulation passed trace horizon {env.horizon:.1f}")
        dt = t - S.now
        if dt > 0:
            busy += dt * sum(1 for j in range(m) if running[j] is not None)
        S.now = t
        while heap and heap[0][0] <= t + EPS:
            _, kind, _, j, tid, tok = heapq.heappop(heap)
            events += 1
            if kind == FINISH:
                if status[tid] != "running" or token[tid] != tok or run_vm[tid] != j:
                    continue
                status[tid] = "done"
                done_t[tid] = S.now
                useful += work[tid]
                running[j] = None
                remaining -= 1
                for c in wf.tasks[tid].children:
                    left[c] -= 1
                    if left[c] == 0:
                        status[c] = "pending"
                        retry_after[c] = S.now
                        pending.add(c)
            elif kind == OUT_END:
                up[j] = True
                if policy.uses_monitor:
                    policy.observe_end(j, S.now)
            elif kind == OUT_START:
                up[j] = False
                observed_outages += 1
                if policy.uses_monitor:
                    policy.observe_start(j, S.now)
                r = running[j]
                if r is not None:
                    lost += (S.now - start_t[r]) * speeds[j]
                    interruptions += 1
                    status[r] = "pending"
                    token[r] += 1
                    run_vm[r] = -1
                    retry_after[r] = S.now + o
                    pending.add(r)
                    running[j] = None
                    push(S.now + o, RETRY, -1, r)
                for q in queue[j]:
                    status[q] = "pending"
                    run_vm[q] = -1
                    retry_after[q] = S.now
                    pending.add(q)
                    released += 1
                queue[j] = []
            # RETRY: nothing to do; dispatch below reconsiders the task
        if events > max_events:
            raise RuntimeError("event budget exceeded")
        start_idle()
        dispatch()

    makespan = max(done_t)
    return {
        "makespan": float(makespan),
        "useful_work": useful,
        "lost_work": lost,
        "interruptions": interruptions,
        "released_queued": released,
        "observed_outages": observed_outages,
        "decisions": decisions,
        "sched_time_s": sched_time,
        "utilization": busy / (m * makespan) if makespan > 0 else 0.0,
        "work_conserved": int(math.isclose(useful, wf.total_work(), rel_tol=1e-9, abs_tol=1e-9)),
        "events": events,
        "decision_log": S.decisions if record else None,
    }
