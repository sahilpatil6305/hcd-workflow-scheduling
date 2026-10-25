# Cell 1 — Imports and configuration
from __future__ import annotations
import heapq
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

MASTER_SEED = 20261022
VM_SPEEDS = [1.00, 1.20, 0.82, 1.45, 0.95, 1.12, 0.72, 1.30]
BASE_FAILURE_RATE = 0.012       # ordinary outages per unit of healthy uptime
REPAIR_TIME = 3.0               # fixed simulated-time repair duration
RESTART_OVERHEAD = 0.25          # simulated dispatch/reinitialization delay after lost work
PRIOR_RATE = BASE_FAILURE_RATE   # common, non-oracle prior for adaptive hazard estimates
SHOCK_VMS = (0, 1, 2)            # VMs hit by the correlated shock
SHOCK_START_FRACTION = 0.40
SHOCK_DURATION_FRACTION = 0.18
PRIOR_EXPOSURE = 30.0            # Gamma-Poisson prior exposure, time units
RISK_WEIGHT = 8.0
ADAPTIVE_WEIGHT_KAPPA = 0.35
MAX_ADAPTIVE_WEIGHT_MULTIPLIER = 2.5
DEADLINE_SLACK_GUARD_FRACTION = 0.15
DEADLINE_WEIGHT = 2.0
WORKLOAD_SIZES = (12, 24, 48)
WORKFLOW_STRUCTURES = ("chain", "fork_join", "random")
FAILURE_MODES = ("stable", "hotspot", "shock")
SCHEDULERS = ("HEFT-ECT", "STATIC-RA-Oracle", "AR-HEFT", "AR-HEFT-NoRisk", "AR-HEFT-NoDeadline")

# Full experiment defaults: start small, validate, then scale.
RUN_MAIN_EXPERIMENT = True
N_SEEDS_MAIN = 100
DEADLINE_FACTOR_MAIN = 1.05
RUN_DEADLINE_SENSITIVITY = False
N_SEEDS_SENSITIVITY = 5
DEADLINE_FACTORS = (0.90, 1.00, 1.05, 1.15, 1.35)

assert len(VM_SPEEDS) >= 3 and all(v > 0 for v in VM_SPEEDS)
assert BASE_FAILURE_RATE >= 0 and REPAIR_TIME > 0
print("Configuration loaded. Schedulers:", SCHEDULERS)

# Cell 2 — Workflow generators and DAG validation
STRUCTURE_CODE = {"chain": 11, "fork_join": 23, "random": 37}

@dataclass
class Task:
    task_id: int
    work: float
    parents: Tuple[int, ...]
    children: Tuple[int, ...] = ()


def make_workflow(structure: str, n_tasks: int, seed: int) -> Dict[int, Task]:
    if structure not in STRUCTURE_CODE:
        raise ValueError(f"Unknown structure: {structure}")
    if n_tasks < 2:
        raise ValueError("n_tasks must be at least 2")
    rng = np.random.default_rng(seed)
    edges = set()
    if structure == "chain":
        edges = {(i, i + 1) for i in range(n_tasks - 1)}
    elif structure == "fork_join":
        # A source, parallel middle layer, and a sink; extra tasks are layered.
        if n_tasks == 2:
            edges = {(0, 1)}
        else:
            middle = list(range(1, n_tasks - 1))
            edges.update((0, j) for j in middle)
            edges.update((j, n_tasks - 1) for j in middle)
    else:
        # Always include a chain backbone to guarantee connectivity; add forward edges.
        edges = {(i, i + 1) for i in range(n_tasks - 1)}
        for i in range(n_tasks):
            for j in range(i + 2, n_tasks):
                if rng.random() < min(0.18, 2.0 / n_tasks):
                    edges.add((i, j))
    parents = {i: [] for i in range(n_tasks)}
    children = {i: [] for i in range(n_tasks)}
    for a, b in sorted(edges):
        children[a].append(b)
        parents[b].append(a)
    works = rng.uniform(4.0, 12.0, size=n_tasks)
    tasks = {
        i: Task(i, float(works[i]), tuple(sorted(parents[i])), tuple(sorted(children[i])))
        for i in range(n_tasks)
    }
    validate_workflow(tasks)
    return tasks


def validate_workflow(tasks: Dict[int, Task]) -> None:
    ids = set(tasks)
    if not ids:
        raise ValueError("Workflow is empty")
    for tid, task in tasks.items():
        if task.task_id != tid or task.work <= 0:
            raise ValueError("Invalid task id or nonpositive work")
        if any(p not in ids or p >= tid for p in task.parents):
            raise ValueError("Workflow must be acyclic with parent IDs preceding children")
        if any(c not in ids or c <= tid for c in task.children):
            raise ValueError("Invalid child edge")
        for p in task.parents:
            if tid not in tasks[p].children:
                raise ValueError("Parent/child edges are inconsistent")
        for c in task.children:
            if tid not in tasks[c].parents:
                raise ValueError("Child/parent edges are inconsistent")


def upward_ranks(tasks: Dict[int, Task], speeds: List[float]) -> Dict[int, float]:
    avg_speed = float(np.mean(speeds))
    ranks = {}
    for tid in sorted(tasks, reverse=True):
        own = tasks[tid].work / avg_speed
        ranks[tid] = own + max((ranks[c] for c in tasks[tid].children), default=0.0)
    return ranks

print("Workflow generators loaded.")

# Cell 2b — Optional generic CSV DAG loader for externally sourced workflows
def load_workflow_csv(task_csv_path, edge_csv_path, id_col="task_id",
                      work_col="work", source_col="source", target_col="target"):
    task_df = pd.read_csv(task_csv_path)
    edge_df = pd.read_csv(edge_csv_path)
    required_task = {id_col, work_col}
    required_edge = {source_col, target_col}
    if not required_task.issubset(task_df.columns):
        raise ValueError(f"Task CSV must contain columns {sorted(required_task)}")
    if not required_edge.issubset(edge_df.columns):
        raise ValueError(f"Edge CSV must contain columns {sorted(required_edge)}")
    ids = task_df[id_col].tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Task identifiers must be unique")
    id_map = {old: i for i, old in enumerate(ids)}
    parents = {i: [] for i in range(len(ids))}
    children = {i: [] for i in range(len(ids))}
    for _, edge in edge_df.iterrows():
        a, b = edge[source_col], edge[target_col]
        if a not in id_map or b not in id_map:
            raise ValueError(f"Edge references unknown task: {(a, b)}")
        ai, bi = id_map[a], id_map[b]
        if ai == bi:
            raise ValueError("Self-loops are invalid")
        children[ai].append(bi)
        parents[bi].append(ai)
    # Topologically reorder nodes so parent IDs precede child IDs.
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(ids)))
    for a in range(len(ids)):
        graph.add_edges_from((a, b) for b in children[a])
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Imported workflow must be a DAG")
    order = list(nx.topological_sort(graph))
    remap = {old_i: new_i for new_i, old_i in enumerate(order)}
    work_by_old = {id_map[row[id_col]]: float(row[work_col]) for _, row in task_df.iterrows()}
    tasks = {}
    for old_i in order:
        new_i = remap[old_i]
        p = tuple(sorted(remap[x] for x in parents[old_i]))
        c = tuple(sorted(remap[x] for x in children[old_i]))
        tasks[new_i] = Task(new_i, work_by_old[old_i], p, c)
    validate_workflow(tasks)
    return tasks

print("Optional external workflow loader ready.")


# Cell 3 — Failure-rate profiles and outage trace generation

def nominal_rate(mode: str, vm_id: int, base_rate: float = BASE_FAILURE_RATE) -> float:
    if mode == "none" or base_rate == 0:
        return 0.0
    if mode == "stable":
        return float(base_rate)
    if mode == "hotspot":
        return float(base_rate * (4.0 if vm_id in (0, 1) else 0.55))
    if mode == "shock":
        return float(base_rate * 0.75)
    raise ValueError(f"Unknown failure mode: {mode}")


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    valid = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not valid:
        return []
    out = [[valid[0][0], valid[0][1]]]
    for start, end in valid[1:]:
        if start <= out[-1][1] + 1e-12:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def generate_outages(
    mode: str,
    seed: int,
    nominal_makespan: float,
    base_rate: float = BASE_FAILURE_RATE,
    n_vms: int = len(VM_SPEEDS),
) -> Dict[int, List[Tuple[float, float]]]:
    if mode not in ("none", *FAILURE_MODES):
        raise ValueError(f"Unknown mode: {mode}")
    if n_vms <= 0 or nominal_makespan <= 0 or base_rate < 0:
        raise ValueError("n_vms and nominal_makespan must be positive; base_rate nonnegative")
    rng = np.random.default_rng(seed)
    horizon = max(10.0 * float(nominal_makespan), float(nominal_makespan) + 2.0 * REPAIR_TIME)
    traces = {i: [] for i in range(n_vms)}
    for vm_id in range(n_vms):
        rate = nominal_rate(mode, vm_id, base_rate)
        if rate <= 0:
            continue
        # Alternating renewal: each ordinary outage begins after healthy uptime.
        t = float(rng.exponential(1.0 / rate))
        while t < horizon:
            end = t + REPAIR_TIME
            traces[vm_id].append((t, end))
            t = end + float(rng.exponential(1.0 / rate))
    if mode == "shock":
        shock_start = SHOCK_START_FRACTION * float(nominal_makespan)
        shock_duration = max(REPAIR_TIME, SHOCK_DURATION_FRACTION * float(nominal_makespan))
        shock_end = shock_start + shock_duration
        for vm_id in SHOCK_VMS:
            if 0 <= vm_id < n_vms:
                traces[vm_id].append((shock_start, shock_end))
    return {i: merge_intervals(v) for i, v in traces.items()}

print("Failure model loaded.")

# Cell 4 — Event-driven simulator and scheduling policies

@dataclass
class RuntimeTask:
    remaining: float
    status: str = "pending"  # pending, running, done
    vm_id: Optional[int] = None
    start_time: Optional[float] = None
    token: int = 0
    completion_time: Optional[float] = None
    retry_after: float = 0.0
    interrupted_at: Optional[float] = None

@dataclass
class VMState:
    down: bool = False
    busy_task: Optional[int] = None
    queue: List[int] = field(default_factory=list)
    last_outage_end: float = 0.0
    failure_count: int = 0
    healthy_exposure: float = 0.0
    busy_time: float = 0.0


def estimated_queue_wait(vm_id, now, tasks, runtime, vm_states, speeds):
    """Projected wait from the running task plus all already queued work."""
    vm = vm_states[vm_id]
    wait = 0.0
    if vm.busy_task is not None:
        current = runtime[vm.busy_task]
        elapsed = max(0.0, now - (current.start_time if current.start_time is not None else now))
        remaining = max(0.0, current.remaining - elapsed * speeds[vm_id])
        wait += remaining / speeds[vm_id]
    wait += sum(runtime[tid].remaining / speeds[vm_id] for tid in vm.queue)
    return wait


def choose_vm(scheduler, task_id, now, tasks, runtime, vm_states, speeds, ranks,
              static_rates, prior_rate, deadline, deadline_weight, risk_weight, avg_speed):
    # Busy VMs remain candidates: their queue delay is included in EFT.
    candidates = [i for i, vm in enumerate(vm_states) if not vm.down]
    if not candidates:
        return None
    task = tasks[task_id]
    tail = max(0.0, ranks[task_id] - task.work / avg_speed)
    deadline_aware = scheduler in ("AR-HEFT", "AR-HEFT-NoRisk")
    use_risk = scheduler in ("STATIC-RA-Oracle", "AR-HEFT", "AR-HEFT-NoDeadline")
    duration = runtime[task_id].remaining / 1.0
    waits = {j: estimated_queue_wait(j, now, tasks, runtime, vm_states, speeds) for j in candidates}
    projected = {j: now + waits[j] + runtime[task_id].remaining / speeds[j] + tail for j in candidates}
    feasible = [j for j in candidates if projected[j] <= deadline] if deadline_aware else candidates
    eligible = feasible if feasible else candidates
    scored = []

    for vm_id in eligible:
        duration = runtime[task_id].remaining / speeds[vm_id]
        start_est = now + waits[vm_id]
        finish = start_est + duration
        score = finish
        if use_risk:
            if scheduler in ("AR-HEFT", "AR-HEFT-NoDeadline"):
                alpha0 = max(prior_rate, 0.0) * PRIOR_EXPOSURE
                beta0 = PRIOR_EXPOSURE
                alpha = alpha0 + vm_states[vm_id].failure_count
                beta = beta0 + vm_states[vm_id].healthy_exposure
                hazard = alpha / beta if beta > 0 else prior_rate
            else:
                hazard = static_rates[vm_id]
            p_fail = -math.expm1(-max(0.0, hazard) * duration)
            expected_penalty = p_fail * (0.5 * duration + RESTART_OVERHEAD)
            effective_risk_weight = risk_weight
            if scheduler in ("AR-HEFT", "AR-HEFT-NoDeadline"):
                total_observed = sum(v.failure_count for v in vm_states)
                total_exposure = sum(v.healthy_exposure for v in vm_states)
                observed_rate = total_observed / max(total_exposure, 1e-9)
                if prior_rate > 0:
                    multiplier = 1.0 + ADAPTIVE_WEIGHT_KAPPA * max(0.0, observed_rate / prior_rate - 1.0)
                    effective_risk_weight *= min(MAX_ADAPTIVE_WEIGHT_MULTIPLIER, multiplier)
                # When the best estimated critical-path completion has little
                # deadline slack, reduce the risk tilt so it cannot dominate QoS.
                if deadline_aware and math.isfinite(deadline):
                    best_projected = min(projected.values())
                    task_slack = max(0.0, deadline - best_projected)
                    slack_scale = min(1.0, task_slack / max(DEADLINE_SLACK_GUARD_FRACTION * deadline, 1e-9))
                    effective_risk_weight *= max(0.10, slack_scale)
            score += effective_risk_weight * expected_penalty
        if deadline_aware and not feasible:
            score += deadline_weight * max(0.0, projected[vm_id] - deadline)
        scored.append((score, vm_id))
    return min(scored, key=lambda x: (x[0], x[1]))[1]


def simulate_workflow(tasks, speeds, outages, scheduler="AR-HEFT", deadline=float("inf"),
                      base_rate=BASE_FAILURE_RATE, risk_weight=RISK_WEIGHT,
                      deadline_weight=DEADLINE_WEIGHT, failure_mode="stable"):
    if scheduler not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {scheduler}")
    validate_workflow(tasks)
    n_vms = len(speeds)
    runtime = {tid: RuntimeTask(task.work) for tid, task in tasks.items()}
    vm_states = [VMState() for _ in speeds]
    ranks = upward_ranks(tasks, speeds)
    static_rates = [nominal_rate(failure_mode, i, base_rate) for i in range(n_vms)]

    # Event tuple: (time, priority, serial, kind, vm_id, task_id, token).
    # Completion is processed before an outage starting at the same timestamp.
    event_q, serial = [], 0
    def push(t, priority, kind, vm_id=-1, task_id=-1, token=-1):
        nonlocal serial
        serial += 1
        heapq.heappush(event_q, (float(t), priority, serial, kind, vm_id, task_id, token))

    generated_intervals = 0
    for vm_id, intervals in outages.items():
        for start, end in intervals:
            generated_intervals += 1
            push(start, 2, "outage_start", vm_id)
            push(end, 1, "outage_end", vm_id)

    now = 0.0
    useful_work = lost_work = executed_work = 0.0
    restarts = interrupted_failures = observed_outages = 0
    recovery_time = scheduling_runtime = 0.0

    def advance_clock(new_time):
        nonlocal now
        dt = max(0.0, new_time - now)
        if dt:
            for vm in vm_states:
                if not vm.down:
                    vm.healthy_exposure += dt
                if vm.busy_task is not None:
                    vm.busy_time += dt
        now = new_time

    def start_queued():
        # Start the head of each queue whenever a VM is healthy and idle.
        for vm_id, vm in enumerate(vm_states):
            if vm.down or vm.busy_task is not None or not vm.queue:
                continue
            tid = vm.queue.pop(0)
            rt = runtime[tid]
            if rt.interrupted_at is not None:
                nonlocal_recovery = max(0.0, now - rt.interrupted_at)
                # Accumulated below through the outer mutable metric holder.
                metric_holder["recovery_time"] += nonlocal_recovery
                rt.interrupted_at = None
            rt.status = "running"
            rt.vm_id = vm_id
            rt.start_time = now
            rt.token += 1
            vm.busy_task = tid
            push(now + rt.remaining / speeds[vm_id], 0, "task_finish", vm_id, tid, rt.token)

    metric_holder = {"recovery_time": 0.0}

    def dispatch():
        nonlocal scheduling_runtime
        while True:
            ready = [
                tid for tid, rt in runtime.items()
                if rt.status == "pending"
                and rt.retry_after <= now + 1e-10
                and all(runtime[p].status == "done" for p in tasks[tid].parents)
            ]
            if not ready or all(vm.down for vm in vm_states):
                break
            deadline_aware = scheduler in ("AR-HEFT", "AR-HEFT-NoRisk")
            if deadline_aware and math.isfinite(deadline):
                def slack_key(tid):
                    candidate_vms = [j for j, vm in enumerate(vm_states) if not vm.down]
                    if not candidate_vms:
                        return (float("inf"), -ranks[tid], tid)
                    best_finish = min(
                        now + estimated_queue_wait(j, now, tasks, runtime, vm_states, speeds)
                        + runtime[tid].remaining / speeds[j]
                        for j in candidate_vms
                    )
                    tail = max(0.0, ranks[tid] - tasks[tid].work / float(np.mean(speeds)))
                    return (deadline - (best_finish + tail), -ranks[tid], tid)
                tid = min(ready, key=slack_key)
            else:
                tid = sorted(ready, key=lambda x: (-ranks[x], x))[0]
            t0 = time.perf_counter()
            vm_id = choose_vm(scheduler, tid, now, tasks, runtime, vm_states, speeds, ranks,
                              static_rates, base_rate, deadline, deadline_weight, risk_weight, float(np.mean(speeds)))
            scheduling_runtime += time.perf_counter() - t0
            if vm_id is None:
                break
            rt = runtime[tid]
            rt.status = "queued"
            rt.vm_id = vm_id
            vm_states[vm_id].queue.append(tid)
        start_queued()

    start_queued()
    dispatch()
    while any(rt.status != "done" for rt in runtime.values()):
        if not event_q:
            unfinished = [tid for tid, rt in runtime.items() if rt.status != "done"]
            raise RuntimeError(f"Simulation deadlock; unfinished tasks: {unfinished}")
        event_time = event_q[0][0]
        advance_clock(event_time)
        same_time = []
        while event_q and abs(event_q[0][0] - event_time) <= 1e-10:
            same_time.append(heapq.heappop(event_q))
        same_time.sort(key=lambda e: (e[1], e[2]))
        for _, _, _, kind, vm_id, tid, token in same_time:
            if kind == "task_finish":
                rt = runtime[tid]
                if rt.status != "running" or rt.token != token or rt.vm_id != vm_id:
                    continue
                progress = min(rt.remaining, max(0.0, (now - rt.start_time) * speeds[vm_id]))
                executed_work += progress
                useful_work += progress
                rt.remaining = max(0.0, rt.remaining - progress)
                rt.status = "done"
                rt.completion_time = now
                rt.vm_id = None
                rt.start_time = None
                vm_states[vm_id].busy_task = None
            elif kind == "outage_end":
                vm_states[vm_id].down = False
                vm_states[vm_id].last_outage_end = now
            elif kind == "outage_start":
                vm_states[vm_id].down = True
                vm_states[vm_id].failure_count += 1
                observed_outages += 1
                interrupted_tid = vm_states[vm_id].busy_task
                if interrupted_tid is not None:
                    rt = runtime[interrupted_tid]
                    if rt.status == "running":
                        progress = min(rt.remaining, max(0.0, (now - rt.start_time) * speeds[vm_id]))
                        executed_work += progress
                        lost_work += progress
                        # True restart semantics: discard partial progress and
                        # re-execute the full task after the restart delay.
                        rt.remaining = tasks[interrupted_tid].work
                        rt.status = "pending"
                        rt.vm_id = None
                        rt.start_time = None
                        rt.token += 1
                        rt.retry_after = now + RESTART_OVERHEAD
                        rt.interrupted_at = now
                        vm_states[vm_id].busy_task = None
                        restarts += 1
                        interrupted_failures += 1
                        push(rt.retry_after, 3, "retry_ready", -1, interrupted_tid)
            elif kind == "retry_ready":
                pass  # dispatch below will reconsider the pending task

        start_queued()
        dispatch()

    recovery_time = metric_holder["recovery_time"]
    makespan = max(rt.completion_time or 0.0 for rt in runtime.values())
    total_busy = sum(vm.busy_time for vm in vm_states)
    utilization = total_busy / (n_vms * makespan) if makespan > 0 else 0.0
    expected_total_work = sum(task.work for task in tasks.values())
    return {
        "makespan": float(makespan),
        "deadline_violation": int(makespan > deadline),
        "deadline": float(deadline),
        "failures": int(interrupted_failures),
        "restarts": int(restarts),
        "observed_outages": int(observed_outages),
        "generated_outage_intervals": int(generated_intervals),
        "useful_work": float(useful_work),
        "lost_work": float(lost_work),
        "executed_work": float(executed_work),
        "recovery_time": float(recovery_time),
        "utilization": float(utilization),
        "scheduling_runtime_seconds": float(scheduling_runtime),
        "workflow_success": int(all(rt.status == "done" for rt in runtime.values())),
        "useful_work_matches": int(math.isclose(useful_work, expected_total_work, rel_tol=1e-7, abs_tol=1e-7)),
    }

print("Event-driven simulator loaded.")


# Cell 5 — Simulator tests (must pass before experiments)

def _test_simulator():
    wf = make_workflow("fork_join", 12, seed=77)
    tr1 = generate_outages("shock", 12, 50.0, 0.0, len(VM_SPEEDS))
    tr2 = generate_outages("shock", 12, 50.0, 0.0, len(VM_SPEEDS))
    assert tr1 == tr2, "Trace generation must be deterministic for a fixed seed."
    assert set(tr1) == set(range(len(VM_SPEEDS)))
    for vm_id, intervals in tr1.items():
        for (a, b), (c, d) in zip(intervals, intervals[1:]):
            assert b <= c + 1e-10, "Intervals must be merged and ordered."

    no_outages = {i: [] for i in range(len(VM_SPEEDS))}
    result = simulate_workflow(wf, VM_SPEEDS, no_outages, "HEFT-ECT",
                               deadline=float("inf"), base_rate=0, failure_mode="none")
    assert result["workflow_success"] == 1
    assert result["makespan"] > 0
    assert result["failures"] == 0 and result["lost_work"] == 0
    assert result["useful_work_matches"] == 1
    assert 0 <= result["utilization"] <= 1.0 + 1e-9
    assert math.isclose(result["useful_work"], sum(t.work for t in wf.values()), rel_tol=1e-8)
    no_failure_ar = simulate_workflow(wf, VM_SPEEDS, no_outages, "AR-HEFT",
                                      deadline=float("inf"), base_rate=0.0, failure_mode="none")
    assert math.isclose(no_failure_ar["makespan"], result["makespan"], rel_tol=1e-9), \
        "With zero failure prior and no outages, AR-HEFT should reduce to the time-based baseline."

    # Controlled interruption: the partial work must be discarded and recomputed.
    tiny = {0: Task(0, 100.0, (), ())}
    controlled = {i: [] for i in range(len(VM_SPEEDS))}
    controlled[0] = [(2.0, 8.0)]
    forced_speeds = [10.0] * len(VM_SPEEDS)
    forced = simulate_workflow(tiny, forced_speeds, controlled,
                               "HEFT-ECT", deadline=float("inf"), base_rate=0)
    assert forced["failures"] >= 1 and forced["restarts"] >= 1
    assert forced["lost_work"] > 0
    assert forced["executed_work"] > forced["useful_work"]
    assert forced["useful_work_matches"] == 1
    assert forced["workflow_success"] == 1
    assert forced["recovery_time"] >= RESTART_OVERHEAD

    # Shock intervals are inserted on the intended VMs at the intended time.
    shock = generate_outages("shock", 9, 100.0, 0.0, len(VM_SPEEDS))
    for vm_id in SHOCK_VMS:
        if vm_id < len(VM_SPEEDS):
            assert any(a <= 40.0 < b for a, b in shock[vm_id])

    # Generated shock must interrupt a running task and account for lost work.
    shock_speeds = [10.0] * len(VM_SPEEDS)
    long_task = {0: Task(0, 1000.0, (), ())}
    generated_shock = generate_outages("shock", 9, 100.0, 0.0, len(VM_SPEEDS))
    shock_run = simulate_workflow(long_task, shock_speeds, generated_shock,
                                  "HEFT-ECT", deadline=float("inf"), base_rate=0.0,
                                  failure_mode="shock")
    assert shock_run["failures"] >= 1
    assert shock_run["lost_work"] > 0
    assert shock_run["useful_work_matches"] == 1
    return True

assert _test_simulator()
print("All corrected simulator tests passed.")


# Cell 6 — Scenario runner

def run_scenario(structure, n_tasks, seed, mode, deadline_factor=DEADLINE_FACTOR_MAIN,
                 base_rate=BASE_FAILURE_RATE, schedulers=SCHEDULERS):
    workload_seed = int(seed + STRUCTURE_CODE[structure] * 1000 + n_tasks * 17)
    tasks = make_workflow(structure, n_tasks, workload_seed)
    no_outages = {i: [] for i in range(len(VM_SPEEDS))}
    ideal = simulate_workflow(tasks, VM_SPEEDS, no_outages, "HEFT-ECT",
                              deadline=float("inf"), base_rate=0.0, failure_mode="none")
    nominal_makespan = ideal["makespan"]
    deadline = nominal_makespan * deadline_factor
    trace_seed = int(seed + STRUCTURE_CODE[structure] + n_tasks * 13 + MASTER_SEED)
    outages = generate_outages(mode, trace_seed, nominal_makespan, base_rate, len(VM_SPEEDS))
    rows = []
    trace_signature = tuple((vm, tuple(outages[vm])) for vm in sorted(outages))
    for scheduler in schedulers:
        result = simulate_workflow(tasks, VM_SPEEDS, outages, scheduler,
                                   deadline=deadline, base_rate=base_rate, failure_mode=mode)
        rows.append({
            "seed": int(seed), "structure": structure, "n_tasks": int(n_tasks),
            "failure_mode": mode, "base_failure_rate": float(base_rate),
            "deadline_factor": float(deadline_factor), "deadline": float(deadline),
            "scheduler": scheduler, "nominal_makespan": nominal_makespan,
            "trace_seed": trace_seed, "outage_trace_id": f"{structure}:{n_tasks}:{mode}:{trace_seed}",
            "trace_interval_count": sum(len(v) for v in outages.values()),
            **result,
        })
    return rows

smoke_rows = run_scenario("fork_join", 12, seed=1, mode="shock", deadline_factor=1.0)
smoke_df = pd.DataFrame(smoke_rows)
print(smoke_df[["scheduler", "makespan", "deadline_violation", "failures", "restarts",
                  "observed_outages", "lost_work", "recovery_time", "utilization"]].round(4))
# The same workload, deadline, and trace must be shared across all scheduler rows.
assert smoke_df["outage_trace_id"].nunique() == 1
assert smoke_df["deadline"].nunique() == 1
assert len(smoke_rows) == len(SCHEDULERS)
assert smoke_df["workflow_success"].all()
print("Smoke test completed.")