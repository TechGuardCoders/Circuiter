"""
Circuiter - GPU partitioning lab.

Slices silicon like MIG does: partition a GPU into slices, enforce per-slice
resource limits, schedule fairly across tenants, then pressure-test the
fairness claims with contention drills.

Runs entirely in-process against a calibrated model of the GB10 partition
table (the real Spark serves production; this lab never touches it). The
scheduler, limit enforcement, and contention dynamics are the real code.

Slice model (MIG-shaped):
- a GPU = N memory/compute slots (GB10: 1 instance profile in production,
  so we model the 2-GPU pair as 8 logical slices)
- a slice = fixed memory cap + share of SM time
- tenants lease slices; oversubscription is where fairness is proven
"""
from __future__ import annotations

import json
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

TICK_S = 1.0
SLICE_MEM_GB = 12          # per logical slice
GPU_TOTAL_SLICES = 8       # 2x DGX Spark pair modeled as 8 slices


@dataclass
class Slice:
    id: int
    tenant: str | None = None
    mem_used_gb: float = 0.0
    sm_busy: float = 0.0        # 0..1 compute load this tick

    @property
    def leased(self) -> bool:
        return self.tenant is not None


@dataclass
class Tenant:
    id: str
    weight: int = 1             # fair-share weight
    max_slices: int = 2         # hard resource limit
    request_queue: deque = field(default_factory=deque)


class Cluster:
    """8 slices, tenants lease up to max_slices, scheduler is weighted-fair."""

    def __init__(self, tenants: list[Tenant]):
        self.slices = [Slice(id=i) for i in range(GPU_TOTAL_SLICES)]
        self.tenants = {t.id: t for t in tenants}
        self.violations: list[str] = []
        self.history: list[dict] = []
        self.completed: dict[str, float] = {}

    def lease(self, tenant_id: str, want: int) -> int:
        t = self.tenants[tenant_id]
        held = sum(1 for s in self.slices if s.tenant == tenant_id)
        granted = 0
        for _ in range(min(want, t.max_slices - held)):
            free = next((s for s in self.slices if not s.leased), None)
            if free is None:
                break
            free.tenant = tenant_id
            granted += 1
        return granted

    def release(self, tenant_id: str, n: int = 1) -> int:
        freed = 0
        for s in self.slices:
            if freed >= n:
                break
            if s.tenant == tenant_id:
                s.tenant, s.mem_used_gb, s.sm_busy = None, 0.0, 0.0
                freed += 1
        return freed

    def submit(self, tenant_id: str, mem_gb: float, sm_demand: float):
        """A request enters the tenant queue if within memory limit, else VIOLATION."""
        t = self.tenants[tenant_id]
        held = [s for s in self.slices if s.tenant == tenant_id]
        free_mem = sum(SLICE_MEM_GB - s.mem_used_gb for s in held)
        if mem_gb > free_mem:
            self.violations.append(f"limit: {tenant_id} asked {mem_gb}GB, had {free_mem}GB free")
            return False
        # RESERVE the memory now on held slices (fill order) - a limit check
        # against memory nobody reserved checks nothing
        need = mem_gb
        for s in held:
            take = min(SLICE_MEM_GB - s.mem_used_gb, need)
            s.mem_used_gb += take
            need -= take
            if need <= 0:
                break
        t.request_queue.append((mem_gb, sm_demand, mem_gb))  # (orig_mem, sm_demand, mem_held)
        return True

    def _release_mem(self, tenant_id: str, mem_gb: float) -> None:
        """Return reserved memory to the tenant's held slices."""
        need = mem_gb
        held = [s for s in self.slices if s.tenant == tenant_id]
        for s in held:
            take = min(s.mem_used_gb, need)
            s.mem_used_gb -= take
            need -= take
            if need <= 0:
                break

    def tick(self) -> dict:
        """One scheduler tick: WFQ slice-time share across tenants.
        Budget = held_slices * weight_i / max_weight (slice-time units).
        Requests carry finite sm_demand; completions counted in units."""
        for tid, t in self.tenants.items():
            held = [s for s in self.slices if s.tenant == tid]
            for s in held:
                s.sm_busy = 0.0
            budget = len(held) * t.weight / max(x.weight for x in self.tenants.values())
            work_done = 0.0
            q = t.request_queue
            remaining = budget
            while q and remaining > 0:
                orig_mem, sm, mem_held = q[0]
                take = min(sm, remaining)
                q.popleft()
                sm -= take
                remaining -= take
                work_done += take
                self.completed[tid] = self.completed.get(tid, 0.0) + take
                if sm <= 1e-9:
                    self._release_mem(tid, mem_held)   # done: give memory back
                else:
                    q.appendleft((orig_mem, sm, mem_held))
            held = [s for s in self.slices if s.tenant == tid]
            if held and work_done > 0:
                per = work_done / len(held)
                for s in held:
                    s.sm_busy = min(1.0, per)
        total_sm = statistics.mean([s.sm_busy for s in self.slices] or [0])
        self.history.append({"t": len(self.history), "mean_sm": round(total_sm, 3),
                             "queues": {tid: len(t.request_queue) for tid, t in self.tenants.items()}})
        return {"mean_sm": total_sm}


def contention_drill(tenants: list[Tenant], ticks: int = 30, seed: int = 3) -> dict:
    """All tenants slam the cluster simultaneously. Fairness = throughput
    proportional to weight, bounded by max_slices."""
    rng = random.Random(seed)
    c = Cluster(tenants)
    for tid, t in c.tenants.items():
        c.lease(tid, t.max_slices)   # everyone grabs their allocation first
    for _ in range(ticks):
        for tid, t in c.tenants.items():
            for _ in range(rng.randint(1, 3)):     # everyone submits eagerly
                c.submit(tid, mem_gb=rng.uniform(1, 4), sm_demand=rng.uniform(0.5, 1.0))
        c.tick()
    # fairness: served ratio between tenants should track weight ratio
    served = {}
    for tid, t in c.tenants.items():
        served[tid] = sum(1 for _ in range(len(t.request_queue)))  # leftover
    return {"cluster": c, "served_left": served}
