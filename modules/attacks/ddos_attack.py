import random
import time
import numpy as np
from typing import Dict, Any, List, Set, Tuple
from loguru import logger

from modules.attacks.base_attack import BaseAttack
from modules.rag_network import DRAGNetwork
from modules.data_types import Datapoint


class DDoSAttack(BaseAttack):
    """
    DDoS (Distributed Denial of Service) attack on a DRAG network.

    Inherits from BaseAttack and implements both required abstract methods:
      - execute()          → runs the attack on the network
      - evaluate_success() → compares before/after metrics to score the attack

    The attack works by:
      1. Selecting target nodes via a configurable strategy
      2. Assigning each target a random overload intensity (0.0–1.0)
      3. Stamping each target with an expiry time for auto-recovery
      4. Writing load values into the network so routing logic degrades realistically
      5. Physically disabling critically overloaded nodes (intensity >= disable_threshold)
    """

    def __init__(
        self,
        attack_ratio: float = 0.3,
        attack_iterations: int = 5,
        strategy: str = "random",
        duration: float = 60.0,
        intensity_range: Tuple[float, float] = (0.5, 1.0),
        disable_threshold: float = 0.9,
        seed: int = 42,
    ):
        """
        Args:
            attack_ratio:      Fraction of nodes to target per iteration (0.0–1.0)
            attack_iterations: How many attack rounds to run inside execute()
            strategy:          Node selection strategy —
                               'random' | 'high_connectivity' | 'high_data' | 'specific'
            duration:          Seconds until attacked nodes auto-recover
            intensity_range:   (min, max) overload factor assigned per node
            disable_threshold: Nodes at or above this intensity are set to None
                               in the live network (hard disable)
            seed:              Random seed for reproducibility
        """
        # Initialise BaseAttack with the attack name
        super().__init__(attack_name="ddos")

        self.attack_ratio = attack_ratio
        self.attack_iterations = attack_iterations
        self.strategy = strategy
        self.duration = duration
        self.intensity_range = intensity_range
        self.disable_threshold = disable_threshold
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Internal tracking (separate from BaseAttack.attack_results)
        self._attacked_nodes: Set[int] = set()
        self._attack_history: List[Dict] = []
        self._node_data: Dict = {}          # shared state across iterations

        logger.info(
            f"DDoSAttack initialised | ratio={attack_ratio} | "
            f"strategy={strategy} | duration={duration}s | "
            f"intensity={intensity_range} | disable_at={disable_threshold}"
        )

    # ------------------------------------------------------------------ #
    #  BaseAttack abstract method 1 — execute()                           #
    # ------------------------------------------------------------------ #

    def execute(self, nodes: List, node_data: Dict, strategy: str = None, duration: float = None, intensity_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Execute one DDoS attack round.

        Called from simulator.py as:
            node_attack.execute(nodes, node_data,
                                strategy=attack_strategy,
                                duration=ddos_duration,
                                intensity_range=intensity_range)

        Args:
            nodes:           List of peer nodes
            node_data:       Shared state dict tracking ddos_targets / node_load
            strategy:        Override self.strategy for this call
            duration:        Override self.duration for this call
            intensity_range: Override self.intensity_range for this call
        """
        # Apply per-call overrides temporarily
        _strategy        = self.strategy
        _duration        = self.duration
        _intensity_range = self.intensity_range

        if strategy is not None:
            self.strategy = strategy
        if duration is not None:
            self.duration = duration
        if intensity_range is not None:
            self.intensity_range = intensity_range

        result = self._run_attack_round(nodes, node_data)

        # Restore originals
        self.strategy        = _strategy
        self.duration        = _duration
        self.intensity_range = _intensity_range

        return result

    # ------------------------------------------------------------------ #
    #  BaseAttack abstract method 2 — evaluate_success()                  #
    # ------------------------------------------------------------------ #

    def evaluate_success(
        self,
        original_metrics: Dict[str, float],
        attacked_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Compare before/after metrics to score how effective the attack was.

        Expects both dicts to contain at least:
          - 'availability_percentage' (float 0–100)
          - 'average_load_intensity'  (float 0–1)

        Optional keys that improve the evaluation if present:
          - 'query_success_rate'   (float 0–1)
          - 'avg_response_time_ms' (float)
          - 'retrieval_precision'  (float 0–1)

        Args:
            original_metrics: Metrics captured BEFORE the attack
            attacked_metrics: Metrics captured AFTER the attack

        Returns:
            Dict with degradation percentages, an overall success score (0–1),
            and a human-readable verdict.
        """
        evaluation = {'attack_type': 'ddos', 'metrics_comparison': {}}

        # --- Core metrics (always present) ---
        avail_drop = (
            original_metrics.get('availability_percentage', 100.0)
            - attacked_metrics.get('availability_percentage', 100.0)
        )
        load_increase = (
            attacked_metrics.get('average_load_intensity', 0.0)
            - original_metrics.get('average_load_intensity', 0.0)
        )

        evaluation['metrics_comparison']['availability_drop_pct'] = round(avail_drop, 2)
        evaluation['metrics_comparison']['load_increase'] = round(load_increase, 3)

        # --- Optional metrics (included when provided) ---
        optional_keys = [
            ('query_success_rate',   True),   # True = lower is worse
            ('avg_response_time_ms', False),  # False = higher is worse
            ('retrieval_precision',  True),
        ]
        for key, lower_is_worse in optional_keys:
            if key in original_metrics and key in attacked_metrics:
                delta = attacked_metrics[key] - original_metrics[key]
                evaluation['metrics_comparison'][f'{key}_delta'] = round(delta, 4)

        # --- Overall success score (0.0 = no effect, 1.0 = total disruption) ---
        # Weighted: availability drop carries 60%, load increase 40%
        avail_score = min(avail_drop / 100.0, 1.0)          # normalise 0–1
        load_score = min(load_increase / 1.0, 1.0)          # already 0–1
        success_score = round(0.6 * avail_score + 0.4 * load_score, 3)

        evaluation['success_score'] = success_score         # 0.0–1.0
        evaluation['availability_drop_pct'] = round(avail_drop, 2)

        # --- Human-readable verdict ---
        if success_score >= 0.8:
            verdict = "CRITICAL — network severely disrupted"
        elif success_score >= 0.5:
            verdict = "HIGH — significant degradation observed"
        elif success_score >= 0.25:
            verdict = "MODERATE — partial disruption"
        else:
            verdict = "LOW — network largely resilient"

        evaluation['verdict'] = verdict
        logger.info(f"Attack evaluation: score={success_score} | {verdict}")

        return evaluation

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _select_target_nodes(self, total_nodes: int, node_data: Dict) -> List[int]:
        num_attack = max(1, int(total_nodes * self.attack_ratio))
        
        # exclude nodes already under DDoS so no slots are wasted on duplicates
        already_targeted = set(node_data.get('ddos_targets', {}).keys())
        available_nodes = [i for i in range(total_nodes) if i not in already_targeted]
        
        if not available_nodes:
            logger.info("All nodes already under DDoS, nothing to target")
            return []
        
        # clamp num_attack to what's actually available
        num_attack = min(num_attack, len(available_nodes))
        
        if self.strategy == "specific" and 'target_peers' in node_data:
            candidates = [i for i in node_data['target_peers'] if i in available_nodes]
            targets = candidates[:num_attack]
        elif self.strategy == "high_connectivity" and 'connectivity' in node_data:
            connectivity = node_data['connectivity']
            targets = sorted(available_nodes, key=lambda i: connectivity.get(i, 0), reverse=True)[:num_attack]
        elif self.strategy == "high_data" and 'data_sizes' in node_data:
            data_sizes = node_data['data_sizes']
            targets = sorted(available_nodes, key=lambda i: data_sizes.get(i, 0), reverse=True)[:num_attack]
        else:
            targets = random.sample(available_nodes, num_attack)

        self._attacked_nodes.update(targets)
        logger.info(f"Selected {len(targets)} target(s) via '{self.strategy}': {targets}")
        return targets

    def _run_attack_round(self, nodes: List, node_data: Dict) -> Dict:
        """
        Core DDoS logic for a single round.
        Stages changes in pending_targets before applying to node_data (safe mutation).
        """
        # Input validation
        if not (0.0 <= self.intensity_range[0] <= self.intensity_range[1] <= 1.0):
            logger.warning(f"Invalid intensity_range {self.intensity_range}, reverting to (0.5, 1.0)")
            self.intensity_range = (0.5, 1.0)

        total_nodes = len(nodes)
        target_nodes = self._select_target_nodes(total_nodes, node_data)
        expiry_time = time.time() + self.duration

        # Stage changes — do NOT touch node_data yet
        pending_targets: Dict[int, Dict] = {}

        for idx in target_nodes:
            if idx >= len(nodes) or nodes[idx] is None:
                logger.warning(f"Node {idx} is None or out of bounds, skipping")
                continue

            if idx in node_data.get('ddos_targets', {}):
                logger.warning(f"Node {idx} already under DDoS, skipping duplicate")
                continue

            intensity = round(random.uniform(*self.intensity_range), 3)
            pending_targets[idx] = {
                'intensity': intensity,
                'expires_at': expiry_time
            }

        # Safe apply — only runs if we got here without exceptions
        node_data.setdefault('ddos_targets', {}).update(pending_targets)
        node_data.setdefault('node_load', {}).update(
            {idx: info['intensity'] for idx, info in pending_targets.items()}
        )

        new_count = len(pending_targets)
        cumulative_count = len(node_data['ddos_targets'])

        result = {
            'attack_type': 'ddos',
            'total_nodes': total_nodes,
            'newly_overloaded': new_count,
            'cumulative_overloaded': cumulative_count,
            'overload_ratio': new_count / total_nodes if total_nodes > 0 else 0.0,
            'target_indices': list(pending_targets.keys()),
            'intensity_per_node': {k: v['intensity'] for k, v in pending_targets.items()},
            'attack_duration_seconds': self.duration,
            'expires_at': expiry_time,
            'targeting_strategy': self.strategy,
        }

        self._attack_history.append(result)
        logger.info(
            f"Round complete | new={new_count} | cumulative={cumulative_count}/{total_nodes}"
        )
        return result

    def _recover_expired_nodes(self, node_data: Dict) -> List[int]:
        """
        Remove nodes from ddos_targets and node_load if their expiry has passed.
        Returns list of recovered node indices.
        """
        now = time.time()
        targets: Dict = node_data.get('ddos_targets', {})
        loads: Dict = node_data.get('node_load', {})

        recovered = [idx for idx, info in targets.items() if info['expires_at'] <= now]

        for idx in recovered:
            targets.pop(idx, None)
            loads.pop(idx, None)

        if recovered:
            logger.info(f"Recovered {len(recovered)} node(s): {recovered}")

        return recovered

    @staticmethod
    def apply_to_network(rag_network, node_attack_results: Dict) -> None:
        """
        Physically set critically overloaded nodes to None in the live network,
        based on intensity maps recorded across all attack iterations.
        Called from simulator.py as: DDoSAttack.apply_to_network(rag_network, results)
        Works with both list-peers and dict-peers networks.
        """
        peers = getattr(rag_network, 'peers', None)
        if peers is None:
            logger.warning("apply_to_network: rag_network has no 'peers' attribute")
            return

        disabled = []
        for iter_result in node_attack_results.get('iterations', []):
            attack_result = iter_result.get('attack_result', {})
            intensity_map: Dict = attack_result.get('intensity_per_node', {})
            for idx, intensity in intensity_map.items():
                if intensity < 0.9:
                    continue
                if isinstance(peers, list) and idx < len(peers) and peers[idx] is not None:
                    peers[idx] = None
                    disabled.append(idx)
                elif isinstance(peers, dict) and idx in peers and peers[idx] is not None:
                    peers[idx] = None
                    disabled.append(idx)

        if disabled:
            logger.info(f"apply_to_network: hard-disabled {len(disabled)} node(s): {disabled}")

    def _evaluate_availability(self, nodes: List, node_data: Dict) -> Dict:
        """
        Snapshot current network health for use in execute() return value
        and as input to evaluate_success().
        """
        total_nodes = len(nodes)
        ddos_targets: Dict = node_data.get('ddos_targets', {})
        node_load: Dict = node_data.get('node_load', {})

        active_nodes = total_nodes - len(ddos_targets)
        avg_load = sum(node_load.values()) / len(node_load) if node_load else 0.0

        return {
            'total_nodes': total_nodes,
            'active_nodes': active_nodes,
            'ddos_overloaded_nodes': len(ddos_targets),
            'availability_percentage': (active_nodes / total_nodes * 100) if total_nodes > 0 else 0.0,
            'average_load_intensity': round(avg_load, 3),
            'load_per_node': dict(node_load),
        }

    # ------------------------------------------------------------------ #
    #  Public utility methods                                              #
    # ------------------------------------------------------------------ #

    def get_attack_summary(self) -> Dict:
        """Return complete attack history and configuration."""
        if not self._attack_history:
            return {'message': 'No attacks executed yet'}

        return {
            'attack_name': self.attack_name,
            'attack_type': 'ddos',
            'attack_ratio': self.attack_ratio,
            'strategy': self.strategy,
            'total_rounds': len(self._attack_history),
            'all_attacked_nodes': list(self._attacked_nodes),
            'attack_history': self._attack_history,
        }

    def reset(self) -> None:
        """Reset all internal state for a fresh experiment run."""
        self._attacked_nodes.clear()
        self._attack_history.clear()
        self._node_data.clear()
        self.attack_results = {}
        logger.info("DDoSAttack state reset")