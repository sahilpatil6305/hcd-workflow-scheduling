"""Workflow representation, WfCommons loading, normalisation and ranks."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wfcommons")


@dataclass(frozen=True)
class Task:
    tid: int
    work: float
    parents: Tuple[int, ...]
    children: Tuple[int, ...]


@dataclass(frozen=True)
class Workflow:
    name: str
    app: str
    tasks: Tuple[Task, ...]          # index == tid, topologically ordered

    @property
    def n(self) -> int:
        return len(self.tasks)

    def total_work(self) -> float:
        return float(sum(t.work for t in self.tasks))


def validate(wf: Workflow) -> None:
    ids = set(range(wf.n))
    for i, t in enumerate(wf.tasks):
        if t.tid != i:
            raise ValueError("task ids must equal positions")
        if not (t.work > 0):
            raise ValueError(f"task {i} has nonpositive work")
        for p in t.parents:
            if p not in ids or p >= i:
                raise ValueError("parents must precede children (topological ids)")
            if i not in wf.tasks[p].children:
                raise ValueError("parent/child inconsistency")
        for c in t.children:
            if c not in ids or c <= i:
                raise ValueError("children must follow parents")
            if i not in wf.tasks[c].parents:
                raise ValueError("child/parent inconsistency")


def from_edges(name: str, app: str, works: Sequence[float], edges: Sequence[Tuple[int, int]]) -> Workflow:
    """Build a workflow from works and edges, re-indexing in a deterministic topological order."""
    n = len(works)
    parents = {i: set() for i in range(n)}
    children = {i: set() for i in range(n)}
    for a, b in edges:
        if a == b:
            raise ValueError("self loop")
        parents[b].add(a)
        children[a].add(b)
    # Kahn with smallest-original-index tie-break -> deterministic order.
    import heapq
    indeg = {i: len(parents[i]) for i in range(n)}
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order: List[int] = []
    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        for c in sorted(children[u]):
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(heap, c)
    if len(order) != n:
        raise ValueError("graph has a cycle")
    new = {old: k for k, old in enumerate(order)}
    tasks = tuple(
        Task(new[old], float(works[old]),
             tuple(sorted(new[p] for p in parents[old])),
             tuple(sorted(new[c] for c in children[old])))
        for old in order
    )
    wf = Workflow(name, app, tasks)
    validate(wf)
    return wf


def load_wfcommons(filename: str, mean_speed: float, min_runtime: float = 1e-3) -> Workflow:
    """Load a pre-extracted WfCommons instance and rescale works.

    Works are scaled so the *mean task takes exactly one time unit on a VM of the
    pool's mean speed*: w_i = runtime_i / mean(runtime) * mean_speed.
    Runtimes below `min_runtime` seconds are raised to it (none occur in the shipped set).
    """
    path = filename if os.path.isabs(filename) else os.path.join(DATA_DIR, filename)
    with open(path) as fh:
        d = json.load(fh)
    ids = [t["id"] for t in d["tasks"]]
    idx = {tid: k for k, tid in enumerate(ids)}
    rts = [max(float(t["runtime"]), min_runtime) for t in d["tasks"]]
    mean_rt = sum(rts) / len(rts)
    works = [r / mean_rt * mean_speed for r in rts]
    edges = [(idx[p], idx[t["id"]]) for t in d["tasks"] for p in t["parents"]]
    return from_edges(d["name"], d["app"], works, edges)


def upward_ranks(wf: Workflow, mean_speed: float) -> List[float]:
    """HEFT upward rank without communication costs."""
    r = [0.0] * wf.n
    for t in reversed(wf.tasks):
        r[t.tid] = t.work / mean_speed + max((r[c] for c in t.children), default=0.0)
    return r


def critical_path_lb(wf: Workflow, max_speed: float) -> float:
    """Critical-path lower bound on the failure-free makespan."""
    longest = [0.0] * wf.n
    for t in wf.tasks:
        longest[t.tid] = t.work / max_speed + max((longest[p] for p in t.parents), default=0.0)
    return max(longest)


def max_width(wf: Workflow) -> int:
    """Largest antichain proxy: maximum number of tasks sharing a topological level."""
    level = [0] * wf.n
    for t in wf.tasks:
        level[t.tid] = 1 + max((level[p] for p in t.parents), default=-1)
    from collections import Counter
    return max(Counter(level).values())
