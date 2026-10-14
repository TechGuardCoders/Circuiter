# Circuiter

**GPU partitioning lab — slices, fair scheduling, hard limits, contention proof.**

Real clusters share silicon. The failure mode isn't the happy path, it's the noisy neighbor at 2am. Circuiter partitions a GPU into MIG-style slices, enforces per-tenant resource limits, schedules weighted-fair, then saturates everything to prove the fairness claims hold under pressure.

Part of the [TechGuardCoders portfolio](https://github.com/orgs/TechGuardCoders/repositories): Cost Peep → Batcher → Spinal → Ball Knowledge → Bastion → Furnace → Truffle → Popcorn → QueuedGPU → **Circuiter** → …

## The model

8 logical slices across the 2-GPU pair (12GB compute-memory each, MIG-shaped), tenants lease up to a hard cap, weighted-fair-queue scheduler divides slice-time by weight.

## Contention drill (measured)

Three tenants, weights 3:2:1, saturating load (demand ≫ budget), 30 ticks:

| Metric | Result |
|---|---|
| SM-units completed (units) | 60.0 / 40.0 / 20.0 |
| **Fairness ratio a/b** | **1.50** (expected 1.50) |
| **Fairness ratio a/c** | **3.00** (expected 3.00) |
| Limit violations caught | 263 (oversubscription rejected, not crashed) |
| Peak mean SM utilization | 0.500 (slices time-share; no starvation) |

Under full contention, throughput tracks the weight ratio *exactly*. That's the fairness proof — measured, not asserted.

## Limit enforcement (measured)

Memory is **reserved at admission** — a limit check against unreserved memory checks nothing:

```
tenant with 1 slice (12GB cap): submit 8GB → ok; submit 8GB more → REJECTED
  violation: "a asked 8GB, had 4.0GB free"
after the request completes: 0.0GB held (memory released)
```

263 violations caught during the drill — every oversubmission attempt rejected and logged.

## Bugs the tests caught (the honest part)

1. **Reservation bug** — the limit check compared against memory nobody had reserved; every request passed. Fixed: admission reserves, completion releases. A limit that only checks but never reserves is a suggestion, not a limit.
2. **WFQ budget formula** — first draft divided by tenant-count instead of max-weight; fairness ratios came out 1.04/2.08 instead of 1.50/3.00. Weighted fairness is arithmetic, not vibes — the test now asserts the exact ratio.
3. **Fairness needs saturation** — under light load everyone keeps up and "fairness" is meaningless (ratio 1.0 by definition). Fair scheduling only differentiates when demand exceeds supply; the drill saturates deliberately.

## Run it

```bash
uv venv --python 3.11 .venv && uv pip install pytest
./.venv/Scripts/python -m pytest tests/ -q
./.venv/Scripts/python scripts/contention_drill.py
```

## What's next

Per-slice cost attribution feeds Cost Peep: cost per slice per tenant, the FinOps layer over shared silicon.

## Fleet mode (multi-node)

`Fleet` routes tenant work across a pool of nodes, each an independent `Cluster` of slices. Per-node health (liveness + free memory + SM load); new work goes to the healthiest node with capacity, a marked-down node is skipped, and routing failures are recorded — never silently dropped. Single-`Cluster` usage is unchanged.

```python
from circuiter import Cluster, Fleet, Node, Tenant

fleet = Fleet([Node(id="n1", cluster=Cluster([Tenant(id="a")])),
               Node(id="n2", cluster=Cluster([Tenant(id="a")]))])
node = fleet.route("a", mem_gb=2.0, sm_demand=0.5)  # healthiest node with capacity
fleet.mark_down("n1")                                # failover: n2 takes over
fleet.tick_all()                                     # advance healthy nodes only
```
