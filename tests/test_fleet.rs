//! 1:1 port of tests/test_fleet.py (multi-node routing, per-node health, failover).

use circuiter::{Cluster, Fleet, Node, SLICE_MEM_GB, Tenant};

fn node(nid: &str, n_slices: usize) -> Node {
    let mut cluster = Cluster::new(vec![
        Tenant::new("a").with_max_slices(n_slices),
        Tenant::new("b").with_max_slices(n_slices),
    ]);
    cluster.lease("a", n_slices); // tenant holds slices so submit can reserve
    Node::new(nid, cluster)
}

#[test]
fn test_routes_to_healthiest_node() {
    // Most free memory wins; tie broken by lowest SM load.
    let mut busy = node("busy", 8);
    let idle = node("idle", 8);
    // load up 'busy': reserve memory so free_mem is lower
    busy.cluster.submit("a", 20.0, 1.0);
    let mut f = Fleet::new(vec![busy, idle]).unwrap();
    let landed = f.route("a", 2.0, 0.5);
    assert_eq!(landed.as_deref(), Some("idle"));
}

#[test]
fn test_unhealthy_node_skipped() {
    let down = node("down", 8);
    let up = node("up", 8);
    let mut f = Fleet::new(vec![down, up]).unwrap();
    f.mark_down("down");
    let landed = f.route("a", 1.0, 0.5);
    assert_eq!(landed.as_deref(), Some("up"));
    assert!(!f.health()["down"].healthy);
}

#[test]
fn test_no_capacity_records_failure() {
    let small = node("small", 8);
    let mut f = Fleet::new(vec![small]).unwrap();
    let landed = f.route("a", SLICE_MEM_GB * 8.0 + 1.0, 0.5);
    assert!(landed.is_none());
    assert_eq!(f.routing_failures.len(), 1);
    assert!(f.routing_failures[0].contains("no healthy node"));
}

#[test]
fn test_failover_after_mark_down() {
    let a = node("a", 8);
    let b = node("b", 8);
    let mut f = Fleet::new(vec![a, b]).unwrap();
    assert!(f.route("a", 1.0, 0.5).is_some());
    f.mark_down("a");
    assert_eq!(f.route("a", 1.0, 0.5).as_deref(), Some("b"));
    f.mark_up("a");
    // 'a' is fresh (more free mem) so it wins again
    assert_eq!(f.route("a", 1.0, 0.5).as_deref(), Some("a"));
}

#[test]
fn test_tick_all_skips_unhealthy() {
    let a = node("a", 8);
    let b = node("b", 8);
    let mut f = Fleet::new(vec![a, b]).unwrap();
    f.mark_down("b");
    let before = f.nodes[1].cluster.history.len();
    let out = f.tick_all();
    assert_eq!(out.len(), 1);
    assert!(out.contains_key("a"));
    assert_eq!(f.nodes[1].cluster.history.len(), before, "unhealthy node not ticked");
}

#[test]
fn test_empty_fleet_rejected() {
    assert!(Fleet::new(vec![]).is_err(), "empty fleet must be an error");
}
