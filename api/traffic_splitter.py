"""Traffic splitting logic for A/B test routing."""

import hashlib
import random
import threading


class TrafficSplitter:
    """Routes incoming requests to model variants based on configurable ratios.

    Supports both random assignment and session-based sticky assignment.
    Thread-safe ratio updates via a reentrant lock.
    """

    def __init__(self, model_a_ratio: float = 0.5) -> None:
        if not 0.0 <= model_a_ratio <= 1.0:
            raise ValueError(
                f"model_a_ratio must be between 0.0 and 1.0, got {model_a_ratio}"
            )
        self._model_a_ratio = model_a_ratio
        self._lock = threading.RLock()

    @property
    def model_a_ratio(self) -> float:
        """Current ratio of traffic routed to Model A."""
        with self._lock:
            return self._model_a_ratio

    @property
    def model_b_ratio(self) -> float:
        """Current ratio of traffic routed to Model B."""
        with self._lock:
            return 1.0 - self._model_a_ratio

    def update_ratio(self, model_a_ratio: float) -> None:
        """Update the traffic split ratio.

        Args:
            model_a_ratio: New ratio for Model A (0.0 to 1.0).

        Raises:
            ValueError: If ratio is outside valid range.
        """
        if not 0.0 <= model_a_ratio <= 1.0:
            raise ValueError(
                f"model_a_ratio must be between 0.0 and 1.0, got {model_a_ratio}"
            )
        with self._lock:
            self._model_a_ratio = model_a_ratio

    def assign_variant(self, session_id: str | None = None) -> str:
        """Assign a model variant to a request.

        If a session_id is provided, the assignment is deterministic based on
        the hash of the session_id, ensuring the same user always sees the
        same variant (sticky assignment). Without a session_id, assignment
        is purely random.

        Args:
            session_id: Optional session identifier for sticky assignment.

        Returns:
            The assigned model variant: 'A' or 'B'.
        """
        with self._lock:
            ratio = self._model_a_ratio

        if session_id is not None:
            hash_value = int(hashlib.sha256(session_id.encode()).hexdigest(), 16)
            normalized = (hash_value % 10000) / 10000.0
            return "A" if normalized < ratio else "B"

        return "A" if random.random() < ratio else "B"
