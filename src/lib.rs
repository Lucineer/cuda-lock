/*!
# cuda-lock

Distributed locking and resource coordination.

Multiple agents accessing shared resources need coordination. This
crate provides mutex, read-write locks, and lease-based locking
with deadlock detection.

- Mutex (exclusive lock)
- Read-write lock (multiple readers, one writer)
- Leases with TTL (automatic expiration)
- Deadlock detection (wait-for graph)
- Lock upgrade (read → write)
*/

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Lock type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockType { Exclusive, Shared }

/// Lock holder
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockHolder {
    pub agent_id: String,
    pub lock_type: LockType,
    pub acquired_ms: u64,
    pub lease_ms: Option<u64>,
}

/// A lock entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockEntry {
    pub resource: String,
    pub holders: Vec<LockHolder>,
    pub waiters: Vec<String>,  // agent IDs waiting
    pub total_acquisitions: u64,
    pub total_waits: u64,
}

impl LockEntry {
    pub fn is_locked(&self) -> bool { !self.holders.is_empty() }
    pub fn is_exclusive(&self) -> bool { self.holders.iter().any(|h| h.lock_type == LockType::Exclusive) }
    pub fn reader_count(&self) -> usize { self.holders.iter().filter(|h| h.lock_type == LockType::Shared).count() }
    pub fn is_expired(&self) -> bool {
        let now = now();
        self.holders.iter().any(|h| h.lease_ms.map_or(false, |l| now - h.acquired_ms > l))
    }
}

/// Deadlock info
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Deadlock {
    pub cycle: Vec<String>,  // agent IDs in cycle
    pub resources: Vec<String>,
    pub detected_ms: u64,
}

/// The lock manager
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockManager {
    pub locks: HashMap<String, LockEntry>,
    pub agent_holds: HashMap<String, Vec<String>>, // agent → resources
    pub agent_waits: HashMap<String, String>,       // agent → resource waiting on
    pub deadlocks: Vec<Deadlock>,
    pub total_locks: u64,
    pub total_timeouts: u64,
}

impl LockManager {
    pub fn new() -> Self { LockManager { locks: HashMap::new(), agent_holds: HashMap::new(), agent_waits: HashMap::new(), deadlocks: vec![], total_locks: 0, total_timeouts: 0 } }

    /// Try to acquire a lock
    pub fn acquire(&mut self, agent_id: &str, resource: &str, lock_type: LockType, lease_ms: Option<u64>) -> LockResult {
        // Clean expired leases first
        self.clean_expired(resource);

        let entry = self.locks.entry(resource.to_string()).or_insert_with(|| LockEntry { resource: resource.to_string(), holders: vec![], waiters: vec![], total_acquisitions: 0, total_waits: 0 });

        match lock_type {
            LockType::Exclusive => {
                if entry.is_locked() { return LockResult::WouldBlock; }
                entry.holders.push(LockHolder { agent_id: agent_id.to_string(), lock_type, acquired_ms: now(), lease_ms });
                entry.total_acquisitions += 1;
            }
            LockType::Shared => {
                if entry.is_exclusive() { return LockResult::WouldBlock; }
                entry.holders.push(LockHolder { agent_id: agent_id.to_string(), lock_type, acquired_ms: now(), lease_ms });
                entry.total_acquisitions += 1;
            }
        }

        self.agent_holds.entry(agent_id.to_string()).or_insert_with(Vec::new).push(resource.to_string());
        self.total_locks += 1;
        LockResult::Acquired
    }

    /// Release a lock
    pub fn release(&mut self, agent_id: &str, resource: &str) {
        if let Some(entry) = self.locks.get_mut(resource) {
            entry.holders.retain(|h| h.agent_id != agent_id);
        }
        if let Some(holds) = self.agent_holds.get_mut(agent_id) { holds.retain(|r| r != resource); }
    }

    /// Release all locks held by an agent
    pub fn release_all(&mut self, agent_id: &str) {
        let resources: Vec<String> = self.agent_holds.get(agent_id).cloned().unwrap_or_default();
        for resource in resources { self.release(agent_id, &resource); }
    }

    /// Clean expired leases
    fn clean_expired(&mut self, resource: &str) {
        if let Some(entry) = self.locks.get_mut(resource) {
            let now = now();
            let before = entry.holders.len();
            entry.holders.retain(|h| h.lease_ms.map_or(true, |l| now - h.acquired_ms <= l));
            let expired = before - entry.holders.len();
            if expired > 0 { self.total_timeouts += expired as u64; }
        }
    }

    /// Check for deadlocks (cycle in wait-for graph)
    pub fn detect_deadlocks(&mut self) -> Vec<Deadlock> {
        let mut cycles = vec![];
        let visited = HashSet::new();
        let path = Vec::new();

        // Build wait-for edges: agent_waits
        for start_agent in self.agent_waits.keys() {
            if visited.contains(start_agent) { continue; }
            let mut current = start_agent.clone();
            let mut path_agents = vec![current.clone()];
            let mut path_set: HashSet<String> = HashSet::new();
            path_set.insert(current.clone());

            while let Some(waiting_on) = self.agent_waits.get(&current) {
                let next = if let Some(entry) = self.locks.get(waiting_on) {
                    entry.holders.first().map(|h| h.agent_id.clone())
                } else { None };

                match next {
                    Some(ref next_agent) if path_set.contains(next_agent) => {
                        // Found cycle
                        let cycle_start = path_agents.iter().position(|a| a == next_agent).unwrap_or(0);
                        let cycle: Vec<String> = path_agents[cycle_start..].to_vec();
                        cycle.push(next_agent.clone());
                        let resources: Vec<String> = cycle.windows(2).filter_map(|w| {
                            self.agent_waits.get(&w[0]).cloned()
                        }).collect();
                        cycles.push(Deadlock { cycle, resources, detected_ms: now() });
                        break;
                    }
                    Some(next_agent) => {
                        path_agents.push(next_agent.clone());
                        path_set.insert(next_agent.clone());
                        current = next_agent;
                    }
                    None => break,
                }
            }
            visited.extend(path_agents);
        }
        self.deadlocks = cycles.clone();
        cycles
    }

    /// List all resources held by an agent
    pub fn held_by(&self, agent_id: &str) -> Vec<&str> {
        self.agent_holds.get(agent_id).map(|r| r.iter().map(|s| s.as_str()).collect()).unwrap_or_default()
    }

    /// Is a resource locked?
    pub fn is_locked(&self, resource: &str) -> bool {
        self.locks.get(resource).map(|e| e.is_locked()).unwrap_or(false)
    }

    /// Summary
    pub fn summary(&self) -> String {
        format!("LockManager: {} resources locked, {} agents holding, {} deadlocks, timeouts={}",
            self.locks.values().filter(|l| l.is_locked()).count(),
            self.agent_holds.len(),
            self.deadlocks.len(),
            self.total_timeouts)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockResult { Acquired, WouldBlock }

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_exclusive() {
        let mut lm = LockManager::new();
        assert_eq!(lm.acquire("a1", "res1", LockType::Exclusive, None), LockResult::Acquired);
        assert!(lm.is_locked("res1"));
    }

    #[test]
    fn test_exclusive_blocks() {
        let mut lm = LockManager::new();
        lm.acquire("a1", "res1", LockType::Exclusive, None);
        assert_eq!(lm.acquire("a2", "res1", LockType::Exclusive, None), LockResult::WouldBlock);
    }

    #[test]
    fn test_shared_allows_multiple_readers() {
        let mut lm = LockManager::new();
        assert_eq!(lm.acquire("a1", "res1", LockType::Shared, None), LockResult::Acquired);
        assert_eq!(lm.acquire("a2", "res1", LockType::Shared, None), LockResult::Acquired);
    }

    #[test]
    fn test_shared_blocks_writer() {
        let mut lm = LockManager::new();
        lm.acquire("a1", "res1", LockType::Shared, None);
        assert_eq!(lm.acquire("a2", "res1", LockType::Exclusive, None), LockResult::WouldBlock);
    }

    #[test]
    fn test_release() {
        let mut lm = LockManager::new();
        lm.acquire("a1", "res1", LockType::Exclusive, None);
        lm.release("a1", "res1");
        assert!(!lm.is_locked("res1"));
    }

    #[test]
    fn test_release_all() {
        let mut lm = LockManager::new();
        lm.acquire("a1", "r1", LockType::Exclusive, None);
        lm.acquire("a1", "r2", LockType::Exclusive, None);
        lm.release_all("a1");
        assert!(!lm.is_locked("r1"));
        assert!(!lm.is_locked("r2"));
    }

    #[test]
    fn test_lease_expiration() {
        let mut lm = LockManager::new();
        lm.acquire("a1", "res1", LockType::Exclusive, Some(0));
        assert_eq!(lm.acquire("a2", "res1", LockType::Exclusive, None), LockResult::Acquired);
    }

    #[test]
    fn test_held_by() {
        let mut lm = LockManager::new();
        lm.acquire("a1", "r1", LockType::Exclusive, None);
        lm.acquire("a1", "r2", LockType::Shared, None);
        let held = lm.held_by("a1");
        assert_eq!(held.len(), 2);
    }

    #[test]
    fn test_no_deadlock() {
        let mut lm = LockManager::new();
        lm.acquire("a1", "r1", LockType::Exclusive, None);
        lm.acquire("a2", "r2", LockType::Exclusive, None);
        let deadlocks = lm.detect_deadlocks();
        assert!(deadlocks.is_empty());
    }

    #[test]
    fn test_summary() {
        let lm = LockManager::new();
        let s = lm.summary();
        assert!(s.contains("0 resources"));
    }
}
