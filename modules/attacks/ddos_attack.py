"""
ddos_attack.py  —  Congestion-based DDoS Attack Model for DRAG Evaluation
==========================================================================
Undergraduate Thesis Implementation
------------------------------------
This module simulates a *congestion-based* Distributed Denial-of-Service (DDoS)
attack against a Distributed Retrieval-Augmented Generation (DRAG) network.

Design Philosophy (Undergraduate Scope)
----------------------------------------
This is intentionally a lightweight, academically reasonable model.
It does NOT simulate:
  - Packet-level traffic    (too low-level for a DRAG thesis)
  - TCP retransmission      (irrelevant at the application layer)
  - Real socket / network   (we model logical peers, not OS sockets)
  - Queueing theory (M/M/1) (appropriate for a networking thesis, not here)
  - Async event loops       (unnecessary complexity)

What it DOES model (sufficient for a DRAG thesis):
  - Attack intensity per peer           → how hard a peer is being hit
  - Drop probability                    → probability a query is lost
  - Load penalty                        → how routing avoids overloaded peers
  - Cascading congestion                → neighbours become partially degraded
  - Timed recovery                      → peers recover after attack expires
  - Reproducible randomness             → seeded RNG throughout

Key Design Invariant
---------------------
Nodes are NEVER physically removed from the network.
An overloaded peer remains reachable but degrades gracefully:
  - It may drop queries (probabilistically).
  - It incurs extra overhead for queries that do pass through.
  - It recovers automatically once the attack duration elapses.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public dataclass returned by intercept_query()
# ---------------------------------------------------------------------------

@dataclass
class InterceptResult:
    """
    Outcome of a single query interception check.

    Attributes
    ----------
    dropped : bool
        True  → the peer dropped this query due to DDoS congestion.
                 The simulator MUST treat this as a failed retrieval
                 (exact_match=0, precision=0, recall=0, f1=0).
        False → query was accepted (may still have overhead).
    extra_hops : int
        Additional logical hops incurred while the peer is congested.
        Contributes to the hop-count metric even when the query succeeds.
    extra_msgs : int
        Extra internal messages (retransmit / NACK overhead).
        Contributes to the message-count metric.
    load_intensity : float
        The attack intensity [0.0, 1.0] on this peer at the time of the call.
        0.0 means the peer is healthy.
    """
    dropped: bool = False
    extra_hops: int = 0
    extra_msgs: int = 0
    load_intensity: float = 0.0


# ---------------------------------------------------------------------------
# Core attack class
# ---------------------------------------------------------------------------

class DDoSAttack:
    """
    Congestion-based DDoS attack model for DRAG network evaluation.

    Parameters
    ----------
    attack_ratio : float
        Fraction of peers targeted per wave.
        E.g. 0.3 → 30 % of peers receive direct DDoS traffic each wave.
    attack_iterations : int
        Number of attack waves. Kept for external callers that loop over
        execute(); the class itself does not iterate internally.
    seed : int
        Master seed for the internal RNG.
        ALL random decisions use self._rng so experiments are reproducible.
    cascade_factor : float
        Fraction of a directly-attacked peer's intensity that spills over
        to its neighbours.  E.g. 0.25 → neighbours receive 25 % of the
        primary intensity.
    max_cascade_intensity : float
        Hard cap on any cascaded intensity so neighbours are never as badly
        hit as directly targeted peers.
    """

    def __init__(
        self,
        attack_ratio: float = 0.3,
        attack_iterations: int = 3,
        seed: int = 42,
        cascade_factor: float = 0.25,
        max_cascade_intensity: float = 0.6,
    ):
        self.attack_ratio          = attack_ratio
        self.attack_iterations     = attack_iterations
        self.seed                  = seed
        self.cascade_factor        = cascade_factor
        self.max_cascade_intensity = max_cascade_intensity

        # ── Seeded RNG ────────────────────────────────────────────────────
        # A single random.Random instance is used for ALL stochastic decisions
        # in this class (target selection, intensity sampling, drop decisions).
        # Using self._rng.random() instead of random.random() ensures that
        # experiments with the same seed produce identical results every run,
        # which is essential for thesis reproducibility.
        self._rng = random.Random(seed)

        self._total_waves: int = 0
        self._cumulative_overloaded: int = 0

    # ------------------------------------------------------------------
    # Public interface — called by simulator.py
    # ------------------------------------------------------------------

    def execute(
        self,
        nodes: List[Any],
        node_data: Dict,
        strategy: str = "random",
        duration: float = 60.0,
        intensity_range: Tuple[float, float] = (0.5, 1.0),
    ) -> Dict:
        """
        Launch one DDoS attack wave against the DRAG network.

        This is the primary entry point used by the simulator.  It selects
        target peers, assigns congestion state, and propagates cascading
        congestion to neighbours.  The node list is NEVER mutated.

        Parameters
        ----------
        nodes : list
            Peer objects from the DRAG network.  Used to determine the total
            peer count and, if peers expose adjacency information, to build
            the neighbour graph.  Never mutated.
        node_data : dict
            Shared mutable state dict.  This call adds / updates:
                ``ddos_targets``    → {peer_idx: intensity}
                ``drop_probability``→ {peer_idx: float in [0, 0.95]}
                ``load_penalty``    → {peer_idx: float in [0, 0.90]}
                ``expires_at``      → {peer_idx: Unix-timestamp float}
        strategy : str
            Target-selection strategy: "random" | "targeted" | "sequential".
        duration : float
            How many seconds until attacked peers automatically recover.
        intensity_range : tuple of (float, float)
            Min and max attack intensity for uniform sampling per target.

        Returns
        -------
        dict
            Wave-level statistics for simulator logging.
        """
        return self._run_attack_round(
            nodes, node_data, strategy, duration, intensity_range
        )

    def intercept_query(self, peer_idx: int, node_data: Dict) -> InterceptResult:
        """
        Decide whether an overloaded peer drops the incoming query.

        Call this in the simulator's query loop *before* invoking the peer's
        retrieval method.  If the peer is not under attack, this is a fast
        no-op (one dict lookup).

        Drop Model
        ----------
        A Bernoulli trial with success probability = drop_probability[peer_idx]:
            P(drop) = drop_probability[peer_idx]   if peer is overloaded
                    = 0.0                           otherwise

        Crucially, self._rng is used for the Bernoulli draw (not the module-
        level random.random()).  This guarantees that every random decision in
        the entire attack pipeline is governed by the same seeded generator,
        making drop outcomes reproducible across runs.

        Overhead Model
        --------------
        Even when a query is NOT dropped, a congested peer incurs overhead
        because it must queue, partially process, and retry internally:
            extra_hops ≈ ceil(intensity × 3)   — retransmission hops
            extra_msgs ≈ ceil(intensity × 5)   — NACK / retry messages

        Evaluation Contract (IMPORTANT)
        --------------------------------
        If `result.dropped` is True, the simulator MUST record this query
        as a failed retrieval with:
            exact_match = 0,  precision = 0,  recall = 0,  f1 = 0
        Do NOT skip the query or omit it from the metric average.
        Including dropped queries in the average is what makes DDoS measurably
        hurt the DRAG evaluation scores compared to the baseline.

        Parameters
        ----------
        peer_idx : int
            Index of the peer that would handle the query.
        node_data : dict
            Shared state dict populated by execute().

        Returns
        -------
        InterceptResult
        """
        ddos_targets = node_data.get("ddos_targets", {})

        if peer_idx not in ddos_targets:
            # Peer is healthy — zero overhead, guaranteed no drop.
            return InterceptResult()

        intensity = ddos_targets[peer_idx]
        drop_prob = node_data.get("drop_probability", {}).get(peer_idx, 0.0)

        # ── Overhead ──────────────────────────────────────────────────────
        # These values exist even when the query ultimately succeeds,
        # because a congested peer still consumes extra resources to queue
        # and forward the request.
        extra_hops = max(1, int(intensity * 3))
        extra_msgs = max(1, int(intensity * 5))

        # ── Drop decision via seeded RNG ──────────────────────────────────
        # FIX: Previously used random.random() (module-level, unseeded),
        # which broke reproducibility.  Now uses self._rng.random() so all
        # random decisions in this class share a single seeded generator.
        dropped = self._rng.random() < drop_prob

        return InterceptResult(
            dropped=dropped,
            extra_hops=extra_hops,
            extra_msgs=extra_msgs,
            load_intensity=intensity,
        )

    def evaluate_success(self, node_data: Dict, total_peers: int) -> Dict:
        """
        Compute high-level attack success metrics after all waves.

        Parameters
        ----------
        node_data : dict
            Shared state dict (updated in-place by execute()).
        total_peers : int
            Total number of peers in the network.

        Returns
        -------
        dict
            Keys: fraction_overloaded, avg_drop_probability, avg_intensity,
                  total_waves_executed, cumulative_overloaded.
        """
        ddos_targets = node_data.get("ddos_targets", {})
        drop_probs   = node_data.get("drop_probability", {})

        num_overloaded = len(ddos_targets)
        avg_drop = (
            sum(drop_probs.values()) / len(drop_probs) if drop_probs else 0.0
        )
        avg_intensity = (
            sum(ddos_targets.values()) / len(ddos_targets)
            if ddos_targets else 0.0
        )

        return {
            "num_overloaded":        num_overloaded,
            "total_peers":           total_peers,
            "fraction_overloaded":   num_overloaded / max(total_peers, 1),
            "avg_drop_probability":  avg_drop,
            "avg_intensity":         avg_intensity,
            "total_waves_executed":  self._total_waves,
            "cumulative_overloaded": self._cumulative_overloaded,
        }

    def get_attack_summary(self) -> Dict:
        """Return a compact summary dict for JSON logging."""
        return {
            "attack_type":           "ddos_congestion",
            "attack_ratio":          self.attack_ratio,
            "attack_iterations":     self.attack_iterations,
            "cascade_factor":        self.cascade_factor,
            "total_waves":           self._total_waves,
            "cumulative_overloaded": self._cumulative_overloaded,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_attack_round(
        self,
        nodes: List[Any],
        node_data: Dict,
        strategy: str,
        duration: float,
        intensity_range: Tuple[float, float],
    ) -> Dict:
        """
        Core wave logic executed once per call to execute().

        Steps
        -----
        1. Initialise node_data sub-dicts on first call.
        2. Select target peer indices via the chosen strategy.
        3. For each target:
               a. Sample attack intensity uniformly from intensity_range.
               b. Derive drop_probability from intensity (capped at 0.95).
               c. Derive load_penalty      from intensity (capped at 0.90).
               d. Record expiry timestamp (now + duration).
        4. Propagate cascading congestion to graph or index neighbours.
        5. Update self._total_waves and self._cumulative_overloaded.
        6. Return wave-level statistics.
        """
        # ── Step 1: Initialise sub-dicts ──────────────────────────────────
        for key in ("ddos_targets", "drop_probability", "load_penalty",
                    "expires_at"):
            node_data.setdefault(key, {})

        num_peers      = len(nodes)
        num_to_attack  = max(1, int(num_peers * self.attack_ratio))

        # ── Step 2: Select target peer indices ────────────────────────────
        all_indices = list(range(num_peers))
        targets = self._select_targets(
            all_indices, num_to_attack, strategy,
            node_data.get("ddos_targets", {})
        )

        now = time.time()
        newly_overloaded    = 0
        intensity_per_node: Dict[int, float] = {}

        # ── Step 3: Assign overload metadata per target ───────────────────
        for idx in targets:
            # Attack intensity: a random value in intensity_range.
            # Higher intensity → more packets hitting the peer → higher drop
            # rate and higher load penalty.  Sampled via seeded RNG.
            intensity = self._rng.uniform(*intensity_range)

            # Drop probability:
            # Derived from intensity with a scaling factor of 0.8.
            # Capped at 0.95 so the peer never becomes a complete black-hole;
            # even heavily attacked peers forward ~5 % of traffic (mimicking
            # partial rate-limiting in real network stacks).
            #   drop_probability = min(0.95, intensity × 0.8)
            drop_prob = min(0.95, intensity * 0.8)

            # Load penalty:
            # Used by load-aware routing to down-weight congested peers in
            # peer selection.  Scaled with factor 0.9 so a full-intensity
            # attack reduces a peer's routing score to ~10 % of its baseline.
            # Capped at 0.90 to preserve at least some routing traffic.
            #   load_penalty = min(0.90, intensity × 0.9)
            load_penalty = min(0.9, intensity * 0.9)

            # If the peer was already overloaded from a previous wave, keep
            # whichever intensity is higher (worst-case accumulation).
            prev_intensity       = node_data["ddos_targets"].get(idx, 0.0)
            effective_intensity  = max(prev_intensity, intensity)

            node_data["ddos_targets"][idx]    = effective_intensity
            node_data["drop_probability"][idx] = min(0.95, effective_intensity * 0.8)
            node_data["load_penalty"][idx]     = min(0.90, effective_intensity * 0.9)
            # expires_at: wall-clock time at which this peer recovers.
            node_data["expires_at"][idx]       = now + duration

            intensity_per_node[idx] = effective_intensity
            if prev_intensity == 0.0:
                newly_overloaded += 1

        # ── Step 4: Cascading congestion propagation ──────────────────────
        #
        # Real DRAG peers propagate congestion to their logical neighbours
        # because an overloaded peer redirects or retransmits queries to
        # adjacent nodes, increasing their load in turn.
        #
        # Strategy: use graph-level adjacency if available, otherwise fall
        # back to index-based ±1 neighbours (a simplified ring topology).
        #
        cascade_count = self._propagate_cascade(
            nodes, node_data, targets, intensity_per_node, now, duration
        )

        # ── Step 5: Update wave counters ──────────────────────────────────
        self._total_waves        += 1
        self._cumulative_overloaded += newly_overloaded

        return {
            "newly_overloaded":    newly_overloaded,
            "cascade_neighbours":  cascade_count,
            "cumulative_overloaded": self._cumulative_overloaded,
            "targets":             list(targets),
            "intensity_per_node":  intensity_per_node,
            # Kept for backward compatibility with simulator logging
            "disabled_nodes":      [],
            "removed_indices":     [],
        }

    def _propagate_cascade(
        self,
        nodes: List[Any],
        node_data: Dict,
        targets: List[int],
        intensity_per_node: Dict[int, float],
        now: float,
        duration: float,
    ) -> int:
        """
        Propagate congestion from directly targeted peers to their neighbours.

        Congestion Propagation Model
        -----------------------------
        When a peer is overloaded it cannot handle all incoming queries.
        Some of those queries are retransmitted or rerouted to neighbouring
        peers, partially overloading them as well.  This is modelled as:

            cascade_intensity = primary_intensity × cascade_factor

        where cascade_factor (default 0.25) represents the fraction of the
        primary load that spills over.  The neighbour's intensity is capped
        at max_cascade_intensity (default 0.6) so that cascade victims are
        always less affected than direct targets.

        Neighbour Detection
        --------------------
        Priority 1 — Graph adjacency:
            If each peer object exposes a ``neighbours`` attribute (a list or
            set of peer indices), those real graph edges are used.  This is
            more accurate because DRAG topologies can be irregular meshes or
            DHT rings where index ±1 is not meaningful.

        Priority 2 — Index-based fallback (simplified ring):
            If no ``neighbours`` attribute is found, we fall back to
            index ±1 (left and right neighbours in a virtual ring).
            This is a deliberate simplification appropriate for an
            undergraduate thesis: it captures the *concept* of cascading
            without requiring the full topology to be reconstructed here.
            Clearly documented so examiners understand the trade-off.

        Parameters
        ----------
        nodes            : list of peer objects
        node_data        : shared state dict (mutated in-place)
        targets          : list of directly attacked peer indices
        intensity_per_node : {idx: effective_intensity} for direct targets
        now              : current Unix timestamp
        duration         : seconds until recovery

        Returns
        -------
        int
            Number of neighbours that received a non-trivial cascade update.
        """
        num_peers     = len(nodes)
        target_set    = set(targets)
        cascade_count = 0

        for idx in targets:
            cascade_intensity = intensity_per_node[idx] * self.cascade_factor

            # ── Determine neighbours ──────────────────────────────────────
            peer_obj = nodes[idx] if idx < num_peers else None

            if peer_obj is not None and hasattr(peer_obj, "neighbours"):
                # Priority 1: real graph neighbours exposed by the peer object.
                # This branch is used when the DRAG network provides an
                # adjacency list, giving topologically accurate propagation.
                neighbour_indices = [
                    n for n in peer_obj.neighbours
                    if isinstance(n, int) and 0 <= n < num_peers
                       and n not in target_set
                ]
            else:
                # Priority 2: simplified index-based ring fallback.
                # Treats the peer list as a ring where each peer's neighbours
                # are the peers immediately before and after it by index.
                # Suitable for an undergraduate thesis when full adjacency
                # information is unavailable.
                neighbour_indices = [
                    n for n in (idx - 1, idx + 1)
                    if 0 <= n < num_peers and n not in target_set
                ]

            # ── Apply cascade to each neighbour ───────────────────────────
            for neighbour in neighbour_indices:
                prev_intensity = node_data["ddos_targets"].get(neighbour, 0.0)

                # New intensity is the higher of the existing congestion and
                # the incoming cascade, capped at max_cascade_intensity.
                new_intensity = min(
                    self.max_cascade_intensity,
                    prev_intensity + cascade_intensity
                )

                if new_intensity > prev_intensity:
                    # Update the neighbour's congestion state.
                    node_data["ddos_targets"][neighbour]     = new_intensity
                    node_data["drop_probability"][neighbour] = min(0.95, new_intensity * 0.8)
                    node_data["load_penalty"][neighbour]     = min(0.90, new_intensity * 0.9)
                    # Cascade victims recover at the same time as the peer
                    # that caused their congestion.
                    node_data["expires_at"][neighbour]       = now + duration
                    cascade_count += 1

        return cascade_count

    def _recover_expired_nodes(self, node_data: Dict) -> List[int]:
        """
        Remove overload metadata for peers whose attack duration has elapsed.

        Recovery Process
        -----------------
        Each peer affected by the DDoS (directly targeted or cascade victim)
        has an ``expires_at`` timestamp.  When the wall clock passes that
        timestamp, the peer is considered recovered: its congestion state
        (intensity, drop_probability, load_penalty) is fully cleared and it
        returns to normal operation.

        This models real-world DDoS mitigation (cloud scrubbing, firewall
        rate-limiting) that typically restores a peer within minutes.
        Without this mechanism, every peer hit in the first wave would remain
        degraded for the entire experiment, which is unrealistically pessimistic.

        Call this once per wave, *before* the next execute() call, so that
        recovered peers can be targeted again if the attacker chooses to.

        Parameters
        ----------
        node_data : dict
            Shared state dict (mutated in-place).

        Returns
        -------
        list of int
            Peer indices that recovered this cycle (for simulator logging).
        """
        now       = time.time()
        recovered: List[int] = []

        for idx in list(node_data.get("expires_at", {}).keys()):
            if now >= node_data["expires_at"][idx]:
                recovered.append(idx)
                for key in ("ddos_targets", "drop_probability",
                            "load_penalty", "expires_at"):
                    node_data.get(key, {}).pop(idx, None)

        return recovered

    def _evaluate_availability(
        self, nodes: List[Any], node_data: Dict
    ) -> Dict:
        """
        Compute availability metrics for the current wave.

        Availability Definition
        -----------------------
        A peer is considered *effectively unavailable* when its
        drop_probability ≥ 0.5, meaning it drops the majority of queries.
        This mirrors a common SLA availability criterion.

        Parameters
        ----------
        nodes     : list of peer objects (used only for total count).
        node_data : shared state dict.

        Returns
        -------
        dict
            Keys: availability_percentage, active_nodes,
                  ddos_overloaded_nodes, effectively_down_nodes,
                  average_load_intensity.
        """
        num_peers    = len(nodes)
        ddos_targets = node_data.get("ddos_targets", {})
        drop_probs   = node_data.get("drop_probability", {})

        effectively_down = sum(1 for p in drop_probs.values() if p >= 0.5)
        active_nodes     = num_peers - effectively_down
        availability_pct = 100.0 * active_nodes / max(num_peers, 1)

        avg_load = (
            sum(ddos_targets.values()) / len(ddos_targets)
            if ddos_targets else 0.0
        )

        return {
            "availability_percentage": availability_pct,
            "active_nodes":            active_nodes,
            "ddos_overloaded_nodes":   len(ddos_targets),
            "effectively_down_nodes":  effectively_down,
            "average_load_intensity":  avg_load,
        }

    # ------------------------------------------------------------------
    # Target selection strategies
    # ------------------------------------------------------------------

    def _select_targets(
        self,
        all_indices: List[int],
        num_to_attack: int,
        strategy: str,
        existing_targets: Dict,
    ) -> List[int]:
        """
        Select which peer indices to attack this wave.

        Strategies
        ----------
        random
            Uniformly random subset via self._rng.sample().
            Good baseline; mimics an unsophisticated botnet.
        targeted
            Prefers peers NOT already overloaded, maximising new damage per
            wave.  Models a smart attacker who avoids redundant flooding.
        sequential
            Round-robin sweep by index.  Models a systematic botnet scan.
            Deterministic given self._total_waves, so no RNG is consumed.

        All strategies use self._rng (or purely deterministic logic) to
        ensure reproducibility when the seed is fixed.
        """
        if strategy == "targeted":
            # Fresh peers first; fall back to already-overloaded ones if needed.
            fresh   = [i for i in all_indices if i not in existing_targets]
            already = [i for i in all_indices if i in existing_targets]
            pool    = fresh + already
            return pool[:num_to_attack]

        elif strategy == "sequential":
            # Deterministic sweep — no RNG needed.
            start   = (self._total_waves * num_to_attack) % len(all_indices)
            indices = [(start + i) % len(all_indices)
                       for i in range(num_to_attack)]
            return indices

        else:  # "random" — default
            return self._rng.sample(
                all_indices, min(num_to_attack, len(all_indices))
            )



