//! Circuiter - GPU partitioning lab (Rust port of `circuiter.py`).
//!
//! Slices silicon like MIG does: partition a GPU into slices, enforce
//! per-slice resource limits, schedule fairly across tenants, then route
//! work across a fleet healthiest-node-first.
//!
//! 1:1 port of the Python reference (same semantics, same test suite).
//! `Fleet::route` returns the node id (`Option<String>`) instead of a Node
//! reference, and an empty fleet is a `Result` error instead of a raise.

use std::collections::{HashMap, VecDeque};

pub const TICK_S: f64 = 1.0;
pub const SLICE_MEM_GB: f64 = 12.0; // per logical slice
pub const GPU_TOTAL_SLICES: usize = 8; // 2x DGX Spark pair modeled as 8 slices

#[derive(Debug, Clone)]
pub struct Slice {
    pub id: usize,
    pub tenant: Option<String>,
    pub mem_used_gb: f64,
    pub sm_busy: f64, // 0..1 compute load this tick
}

impl Slice {
    fn new(id: usize) -> Self {
        Slice { id, tenant: None, mem_used_gb: 0.0, sm_busy: 0.0 }
    }

    pub fn leased(&self) -> bool {
        self.tenant.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct Tenant {
    pub id: String,
    pub weight: u32,     // fair-share weight
    pub max_slices: usize, // hard resource limit
    // (orig_mem, sm_demand, mem_held)
    pub request_queue: VecDeque<(f64, f64, f64)>,
}

impl Tenant {
    pub fn new(id: &str) -> Self {
        Tenant { id: id.to_string(), weight: 1, max_slices: 2, request_queue: VecDeque::new() }
    }

    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_max_slices(mut self, max_slices: usize) -> Self {
        self.max_slices = max_slices;
        self
    }
}

#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub t: usize,
    pub mean_sm: f64,
    pub queues: HashMap<String, usize>,
}

/// 8 slices, tenants lease up to max_slices, scheduler is weighted-fair.
pub struct Cluster {
    pub slices: Vec<Slice>,
    pub tenants: HashMap<String, Tenant>,
    pub violations: Vec<String>,
    pub history: Vec<HistoryEntry>,
    pub completed: HashMap<String, f64>,
}

impl Cluster {
    pub fn new(tenants: Vec<Tenant>) -> Self {
        Cluster {
            slices: (0..GPU_TOTAL_SLICES).map(Slice::new).collect(),
            tenants: tenants.into_iter().map(|t| (t.id.clone(), t)).collect(),
            violations: Vec::new(),
            history: Vec::new(),
            completed: HashMap::new(),
        }
    }

    pub fn lease(&mut self, tenant_id: &str, want: usize) -> usize {
        let max_slices = self.tenants[tenant_id].max_slices;
        let held = self.slices.iter().filter(|s| s.tenant.as_deref() == Some(tenant_id)).count();
        let mut granted = 0;
        for _ in 0..want.min(max_slices.saturating_sub(held)) {
            let free = match self.slices.iter_mut().find(|s| !s.leased()) {
                Some(s) => s,
                None => break,
            };
            free.tenant = Some(tenant_id.to_string());
            granted += 1;
        }
        granted
    }

    pub fn release(&mut self, tenant_id: &str, n: usize) -> usize {
        let mut freed = 0;
        for s in self.slices.iter_mut() {
            if freed >= n {
                break;
            }
            if s.tenant.as_deref() == Some(tenant_id) {
                s.tenant = None;
                s.mem_used_gb = 0.0;
                s.sm_busy = 0.0;
                freed += 1;
            }
        }
        freed
    }

    /// A request enters the tenant queue if within memory limit, else VIOLATION.
    pub fn submit(&mut self, tenant_id: &str, mem_gb: f64, sm_demand: f64) -> bool {
        let held_ids: Vec<usize> = self
            .slices
            .iter()
            .filter(|s| s.tenant.as_deref() == Some(tenant_id))
            .map(|s| s.id)
            .collect();
        let free_mem: f64 = held_ids
            .iter()
            .map(|&id| SLICE_MEM_GB - self.slices[id].mem_used_gb)
            .sum();
        if mem_gb > free_mem {
            self.violations.push(format!(
                "limit: {} asked {}GB, had {}GB free",
                tenant_id, mem_gb, free_mem
            ));
            return false;
        }
        // RESERVE the memory now on held slices (fill order) - a limit check
        // against memory nobody reserved checks nothing
        let mut need = mem_gb;
        for &id in &held_ids {
            let s = &mut self.slices[id];
            let take = (SLICE_MEM_GB - s.mem_used_gb).min(need);
            s.mem_used_gb += take;
            need -= take;
            if need <= 0.0 {
                break;
            }
        }
        let t = self.tenants.get_mut(tenant_id).unwrap();
        t.request_queue.push_back((mem_gb, sm_demand, mem_gb));
        true
    }

    /// Return reserved memory to the tenant's held slices.
    fn release_mem(&mut self, tenant_id: &str, mem_gb: f64) {
        let mut need = mem_gb;
        let held_ids: Vec<usize> = self
            .slices
            .iter()
            .filter(|s| s.tenant.as_deref() == Some(tenant_id))
            .map(|s| s.id)
            .collect();
        for &id in &held_ids {
            let s = &mut self.slices[id];
            let take = s.mem_used_gb.min(need);
            s.mem_used_gb -= take;
            need -= take;
            if need <= 0.0 {
                break;
            }
        }
    }

    /// One scheduler tick: WFQ slice-time share across tenants.
    /// Budget = held_slices * weight_i / max_weight (slice-time units).
    /// Requests carry finite sm_demand; completions counted in units.
    pub fn tick(&mut self) -> f64 {
        let max_weight = self.tenants.values().map(|t| t.weight).max().unwrap_or(1) as f64;
        let tenant_ids: Vec<String> = self.tenants.keys().cloned().collect();
        for tid in &tenant_ids {
            let held_ids: Vec<usize> = self
                .slices
                .iter()
                .filter(|s| s.tenant.as_deref() == Some(tid.as_str()))
                .map(|s| s.id)
                .collect();
            for &id in &held_ids {
                self.slices[id].sm_busy = 0.0;
            }
            let weight = self.tenants[tid].weight as f64;
            let held_count = held_ids.len();
            let budget = held_count as f64 * weight / max_weight;
            let mut work_done = 0.0;
            let mut remaining = budget;
            let mut mem_releases: Vec<f64> = Vec::new();
            {
                let q = &mut self.tenants.get_mut(tid).unwrap().request_queue;
                while !q.is_empty() && remaining > 0.0 {
                    let (orig_mem, mut sm, mem_held) = *q.front().unwrap();
                    let take = sm.min(remaining);
                    q.pop_front();
                    sm -= take;
                    remaining -= take;
                    work_done += take;
                    if sm <= 1e-9 {
                        mem_releases.push(mem_held); // done: give memory back
                    } else {
                        q.push_front((orig_mem, sm, mem_held));
                    }
                }
            }
            *self.completed.entry(tid.clone()).or_insert(0.0) += work_done;
            for mem_held in mem_releases {
                self.release_mem(tid, mem_held);
            }
            if !held_ids.is_empty() && work_done > 0.0 {
                let per = work_done / held_ids.len() as f64;
                for &id in &held_ids {
                    self.slices[id].sm_busy = per.min(1.0);
                }
            }
        }
        let total_sm = self.slices.iter().map(|s| s.sm_busy).sum::<f64>() / self.slices.len() as f64;
        let queues: HashMap<String, usize> = self
            .tenants
            .iter()
            .map(|(tid, t)| (tid.clone(), t.request_queue.len()))
            .collect();
        self.history.push(HistoryEntry { t: self.history.len(), mean_sm: (total_sm * 1000.0).round() / 1000.0, queues });
        total_sm
    }
}

// ---------------------------------------------------------------- fleet
//
// Fleet mode: multiple compute nodes, each an independent Cluster of slices.
// Each node reports health (mean SM utilization + liveness); new work routes
// to the healthiest node with capacity. A node that fails (or is marked down)
// stops receiving work; callers get a routing failure they can retry elsewhere.

pub struct Node {
    pub id: String,
    pub cluster: Cluster,
    pub healthy: bool,
}

impl Node {
    pub fn new(id: &str, cluster: Cluster) -> Self {
        Node { id: id.to_string(), cluster, healthy: true }
    }

    /// Mean SM utilization across this node's slices (0..1).
    pub fn load(&self) -> f64 {
        self.cluster.slices.iter().map(|s| s.sm_busy).sum::<f64>() / self.cluster.slices.len() as f64
    }

    pub fn free_mem_gb(&self) -> f64 {
        self.cluster.slices.iter().map(|s| SLICE_MEM_GB - s.mem_used_gb).sum()
    }
}

#[derive(Debug, Clone)]
pub struct NodeHealth {
    pub healthy: bool,
    pub load: f64,
    pub free_mem_gb: f64,
}

/// Route tenant work across a pool of nodes, healthiest first.
///
/// Selection: among healthy nodes, prefer the one with the most free memory,
/// tie-broken by lowest SM load (first node wins exact ties, matching the
/// Python `max` over insertion order). Unhealthy nodes are skipped entirely.
pub struct Fleet {
    pub nodes: Vec<Node>,
    pub routing_failures: Vec<String>,
}

impl Fleet {
    pub fn new(nodes: Vec<Node>) -> Result<Self, String> {
        if nodes.is_empty() {
            return Err("fleet needs at least one node".to_string());
        }
        Ok(Fleet { nodes, routing_failures: Vec::new() })
    }

    pub fn mark_down(&mut self, node_id: &str) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == node_id) {
            n.healthy = false;
        }
    }

    pub fn mark_up(&mut self, node_id: &str) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == node_id) {
            n.healthy = true;
        }
    }

    pub fn health(&self) -> HashMap<String, NodeHealth> {
        self.nodes
            .iter()
            .map(|n| {
                let load = (n.load() * 1000.0).round() / 1000.0;
                let free = (n.free_mem_gb() * 10.0).round() / 10.0;
                (n.id.clone(), NodeHealth { healthy: n.healthy, load, free_mem_gb: free })
            })
            .collect()
    }

    fn pick(&self, mem_gb: f64) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_key = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (i, n) in self.nodes.iter().enumerate() {
            if !n.healthy || n.free_mem_gb() < mem_gb {
                continue;
            }
            let key = (n.free_mem_gb(), -n.load());
            if best.is_none() || key > best_key {
                best = Some(i);
                best_key = key;
            }
        }
        best
    }

    /// Submit work to the healthiest node with capacity. Returns the id of the
    /// node it landed on, or None (and records a routing failure) if no healthy
    /// node can take it.
    pub fn route(&mut self, tenant_id: &str, mem_gb: f64, sm_demand: f64) -> Option<String> {
        let idx = match self.pick(mem_gb) {
            Some(i) => i,
            None => {
                self.routing_failures.push(format!(
                    "route: {} asked {}GB, no healthy node with capacity",
                    tenant_id, mem_gb
                ));
                return None;
            }
        };
        let node = &mut self.nodes[idx];
        if !node.cluster.submit(tenant_id, mem_gb, sm_demand) {
            // node-level limit violation: tenant over its cap there
            self.routing_failures
                .push(format!("route: {} rejected by {} (limit)", tenant_id, node.id));
            return None;
        }
        Some(node.id.clone())
    }

    /// Tick every healthy node; unhealthy nodes are not advanced.
    pub fn tick_all(&mut self) -> HashMap<String, f64> {
        let mut out = HashMap::new();
        for n in self.nodes.iter_mut() {
            if n.healthy {
                let mean_sm = n.cluster.tick();
                out.insert(n.id.clone(), mean_sm);
            }
        }
        out
    }
}

/// All tenants slam the cluster simultaneously. Fairness = throughput
/// proportional to weight, bounded by max_slices. (Deterministic LCG stands in
/// for Python's seeded `random.Random`.)
pub fn contention_drill(tenants: Vec<Tenant>, ticks: usize, seed: u64) -> (Cluster, HashMap<String, usize>) {
    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state >> 33
    }
    let mut rng = seed;
    let mut c = Cluster::new(tenants);
    let ids: Vec<String> = c.tenants.keys().cloned().collect();
    for tid in &ids {
        let max_slices = c.tenants[tid].max_slices;
        c.lease(tid, max_slices); // everyone grabs their allocation first
    }
    for _ in 0..ticks {
        for tid in &ids {
            let n = 1 + (lcg(&mut rng) % 3) as usize; // everyone submits eagerly
            for _ in 0..n {
                let mem = 1.0 + (lcg(&mut rng) % 1000) as f64 / 1000.0 * 3.0;
                let sm = 0.5 + (lcg(&mut rng) % 1000) as f64 / 1000.0 * 0.5;
                c.submit(tid, mem, sm);
            }
        }
        c.tick();
    }
    let served: HashMap<String, usize> = ids
        .iter()
        .map(|tid| (tid.clone(), c.tenants[tid].request_queue.len()))
        .collect();
    (c, served)
}
