"""Fleet tests: multi-node routing, per-node health, failover. No network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuiter import Cluster, Fleet, Node, Tenant, SLICE_MEM_GB  # noqa: E402


def _node(nid: str, n_slices: int = 8) -> Node:
    n = Node(id=nid, cluster=Cluster([Tenant(id="a", max_slices=n_slices),
                                      Tenant(id="b", max_slices=n_slices)]))
    n.cluster.lease("a", n_slices)  # tenant holds slices so submit can reserve
    return n


def test_routes_to_healthiest_node():
    """Most free memory wins; tie broken by lowest SM load."""
    busy = _node("busy")
    idle = _node("idle")
    # load up 'busy': reserve memory so free_mem is lower
    busy.cluster.submit("a", mem_gb=20.0, sm_demand=1.0)
    f = Fleet([busy, idle])
    landed = f.route("a", mem_gb=2.0, sm_demand=0.5)
    assert landed is not None and landed.id == "idle"


def test_unhealthy_node_skipped():
    down = _node("down")
    up = _node("up")
    f = Fleet([down, up])
    f.mark_down("down")
    landed = f.route("a", mem_gb=1.0, sm_demand=0.5)
    assert landed is not None and landed.id == "up"
    assert f.health()["down"]["healthy"] is False


def test_no_capacity_records_failure():
    small = _node("small")
    f = Fleet([small])
    landed = f.route("a", mem_gb=SLICE_MEM_GB * 8 + 1, sm_demand=0.5)
    assert landed is None
    assert len(f.routing_failures) == 1 and "no healthy node" in f.routing_failures[0]


def test_failover_after_mark_down():
    a = _node("a")
    b = _node("b")
    f = Fleet([a, b])
    assert f.route("a", mem_gb=1.0, sm_demand=0.5).id in ("a", "b")
    f.mark_down("a")
    assert f.route("a", mem_gb=1.0, sm_demand=0.5).id == "b"
    f.mark_up("a")
    # 'a' is fresh (more free mem) so it wins again
    assert f.route("a", mem_gb=1.0, sm_demand=0.5).id == "a"


def test_tick_all_skips_unhealthy():
    a = _node("a")
    b = _node("b")
    f = Fleet([a, b])
    f.mark_down("b")
    before = len(b.cluster.history)
    out = f.tick_all()
    assert set(out) == {"a"}
    assert len(b.cluster.history) == before, "unhealthy node not ticked"


def test_empty_fleet_rejected():
    try:
        Fleet([])
        raise AssertionError("empty fleet must raise")
    except ValueError:
        pass
