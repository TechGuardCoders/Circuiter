"""Offline tests: limits, fairness math, slice accounting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuiter import Cluster, Tenant, Slice, SLICE_MEM_GB, GPU_TOTAL_SLICES  # noqa: E402


def test_lease_respects_hard_cap():
    c = Cluster([Tenant(id="a", max_slices=2)])
    assert c.lease("a", 5) == 2, "grants only max_slices even if asked for more"
    assert c.lease("a", 1) == 0, "cap reached"


def test_lease_fails_when_full():
    c = Cluster([Tenant(id="a", max_slices=8), Tenant(id="b", max_slices=1)])
    assert c.lease("a", 8) == 8
    assert c.lease("b", 1) == 0, "no free slice left"
    c.release("a", 1)
    assert c.lease("b", 1) == 1


def test_memory_limit_enforced():
    c = Cluster([Tenant(id="a", max_slices=2)])
    c.lease("a", 2)  # 24GB total
    assert c.submit("a", mem_gb=25.0, sm_demand=1.0) is False, "over memory limit rejected"
    assert len(c.violations) == 1
    assert c.submit("a", mem_gb=20.0, sm_demand=1.0) is True


def test_weighted_fair_share():
    """Weight 3:1 -> weighted budget split 3/4 vs 1/4 (same slice count)."""
    c = Cluster([Tenant(id="a", weight=3, max_slices=1),
                 Tenant(id="b", weight=1, max_slices=1)])
    c.lease("a", 1); c.lease("b", 1)
    for _ in range(3):
        c.submit("a", mem_gb=1, sm_demand=10)  # always hungry
        c.submit("b", mem_gb=1, sm_demand=10)
    c.tick()
    # a has 3x the slice-time budget -> completes ~3x the units (items are
    # partial-completion; measure units, not queue counts)
    a, b = c.completed["a"], c.completed["b"]
    assert abs(a / b - 3.0) < 0.01, f"WFQ: a/b units {a}/{b} should be ~3"


def test_no_slice_leak_on_release():
    c = Cluster([Tenant(id="a", max_slices=3)])
    c.lease("a", 3)
    c.release("a", 2)
    free = sum(1 for s in c.slices if not s.leased)
    assert free == 7  # 8 total, 1 still held
    held_mem = sum(s.mem_used_gb for s in c.slices)
    assert held_mem == 0, "released slices reset"


def test_slice_count_constant():
    c = Cluster([Tenant(id="a", max_slices=8), Tenant(id="b", max_slices=8)])
    c.lease("a", 5); c.lease("b", 5)  # only 8 exist
    held = sum(1 for s in c.slices if s.leased)
    assert held == 8 == GPU_TOTAL_SLICES
