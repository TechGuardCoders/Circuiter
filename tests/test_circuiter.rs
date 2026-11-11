//! 1:1 port of tests/test_circuiter.py (limits, fairness math, slice accounting).

use circuiter::{Cluster, GPU_TOTAL_SLICES, Tenant};

#[test]
fn test_lease_respects_hard_cap() {
    let mut c = Cluster::new(vec![Tenant::new("a").with_max_slices(2)]);
    assert_eq!(c.lease("a", 5), 2, "grants only max_slices even if asked for more");
    assert_eq!(c.lease("a", 1), 0, "cap reached");
}

#[test]
fn test_lease_fails_when_full() {
    let mut c = Cluster::new(vec![Tenant::new("a").with_max_slices(8), Tenant::new("b").with_max_slices(1)]);
    assert_eq!(c.lease("a", 8), 8);
    assert_eq!(c.lease("b", 1), 0, "no free slice left");
    c.release("a", 1);
    assert_eq!(c.lease("b", 1), 1);
}

#[test]
fn test_memory_limit_enforced() {
    let mut c = Cluster::new(vec![Tenant::new("a").with_max_slices(2)]);
    c.lease("a", 2); // 24GB total
    assert!(!c.submit("a", 25.0, 1.0), "over memory limit rejected");
    assert_eq!(c.violations.len(), 1);
    assert!(c.submit("a", 20.0, 1.0));
}

#[test]
fn test_weighted_fair_share() {
    // Weight 3:1 -> weighted budget split 3/4 vs 1/4 (same slice count).
    let mut c = Cluster::new(vec![
        Tenant::new("a").with_weight(3).with_max_slices(1),
        Tenant::new("b").with_max_slices(1),
    ]);
    c.lease("a", 1);
    c.lease("b", 1);
    for _ in 0..3 {
        c.submit("a", 1.0, 10.0); // always hungry
        c.submit("b", 1.0, 10.0);
    }
    c.tick();
    // a has 3x the slice-time budget -> completes ~3x the units (items are
    // partial-completion; measure units, not queue counts)
    let a = c.completed["a"];
    let b = c.completed["b"];
    assert!((a / b - 3.0).abs() < 0.01, "WFQ: a/b units {}/{} should be ~3", a, b);
}

#[test]
fn test_no_slice_leak_on_release() {
    let mut c = Cluster::new(vec![Tenant::new("a").with_max_slices(3)]);
    c.lease("a", 3);
    c.release("a", 2);
    let free = c.slices.iter().filter(|s| !s.leased()).count();
    assert_eq!(free, 7, "8 total, 1 still held");
    let held_mem: f64 = c.slices.iter().map(|s| s.mem_used_gb).sum();
    assert_eq!(held_mem, 0.0, "released slices reset");
}

#[test]
fn test_slice_count_constant() {
    let mut c = Cluster::new(vec![Tenant::new("a").with_max_slices(8), Tenant::new("b").with_max_slices(8)]);
    c.lease("a", 5);
    c.lease("b", 5); // only 8 exist
    let held = c.slices.iter().filter(|s| s.leased()).count();
    assert_eq!(held, 8);
    assert_eq!(held, GPU_TOTAL_SLICES);
}
