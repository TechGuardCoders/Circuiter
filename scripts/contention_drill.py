"""Contention drill runner - the fairness proof, measured."""
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuiter import Cluster, Tenant, contention_drill  # noqa: E402


def measure_fairness(tenants, ticks=30):
    """Track completed work per tenant across the drill."""
    rng_seed = 3
    result = contention_drill(tenants, ticks=ticks, seed=rng_seed)
    c = result["cluster"]
    # count actually-completed work from history queues is indirect; use sm_busy integral
    integral = {tid: 0.0 for tid in c.tenants}
    # recompute: rerun with per-tick capture
    return c, integral


def main():
    print("=== Circuiter contention drill ===")
    print("3 tenants, weights 3:2:1, hard cap 2 slices each, 8 slices total\n")

    tenants = [Tenant(id="tenant-a", weight=3, max_slices=2),
               Tenant(id="tenant-b", weight=2, max_slices=2),
               Tenant(id="tenant-c", weight=1, max_slices=2)]

    c = Cluster(tenants)
    leases = {t.id: c.lease(t.id, t.max_slices) for t in tenants}
    print(f"leases granted: {leases} (cap 2 each, 8 slices)")

    import random
    rng = random.Random(3)
    completed = {tid: 0.0 for tid in c.tenants}
    submitted = {tid: 0 for tid in c.tenants}

    for tick in range(30):
        for tid in c.tenants:
            for _ in range(rng.randint(4, 6)):     # saturate: demand >> budget
                ok = c.submit(tid, mem_gb=rng.uniform(1, 3), sm_demand=rng.uniform(0.5, 1.0))
                if ok:
                    submitted[tid] += 1
        c.tick()

    print(f"\nrequests submitted: {submitted}")
    units = {tid: round(c.completed.get(tid, 0.0), 1) for tid in c.tenants}
    print(f"sm-units completed (30 ticks): {units}")

    a, b, cc = units["tenant-a"], units["tenant-b"], units["tenant-c"]
    ratio_ab = a / max(b, 1e-9)
    ratio_ac = a / max(cc, 1e-9)
    print(f"\nfairness ratios (units): a/b = {ratio_ab:.2f} (expect ~1.5), a/c = {ratio_ac:.2f} (expect ~3.0)")

    violations = c.violations
    print(f"limit violations caught: {len(violations)}")
    if violations[:2]:
        print("  sample:", violations[0])

    peak = max(h["mean_sm"] for h in c.history)
    print(f"peak mean SM utilization across contention: {peak:.3f}")

    Path("benchmarks").mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    Path(f"benchmarks/circuiter_{stamp}.json").write_text(json.dumps({
        "submitted": submitted, "units": units,
        "ratios": {"a/b": round(ratio_ab, 2), "a/c": round(ratio_ac, 2)},
        "violations": violations, "peak_mean_sm": peak,
        "history": c.history}, indent=2))
    print(f"saved benchmarks/circuiter_{stamp}.json")


if __name__ == "__main__":
    main()
