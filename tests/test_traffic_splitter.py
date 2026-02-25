"""Unit tests for the traffic splitting logic."""

import pytest

from api.traffic_splitter import TrafficSplitter


class TestTrafficSplitter:
    """Tests for the TrafficSplitter class."""

    def test_default_ratio_is_fifty_fifty(self):
        splitter = TrafficSplitter()
        assert splitter.model_a_ratio == 0.5
        assert splitter.model_b_ratio == 0.5

    def test_custom_ratio(self):
        splitter = TrafficSplitter(model_a_ratio=0.7)
        assert splitter.model_a_ratio == 0.7
        assert abs(splitter.model_b_ratio - 0.3) < 1e-9

    def test_invalid_ratio_raises_error(self):
        with pytest.raises(ValueError):
            TrafficSplitter(model_a_ratio=1.5)

    def test_negative_ratio_raises_error(self):
        with pytest.raises(ValueError):
            TrafficSplitter(model_a_ratio=-0.1)

    def test_update_ratio(self):
        splitter = TrafficSplitter(model_a_ratio=0.5)
        splitter.update_ratio(0.8)
        assert splitter.model_a_ratio == 0.8
        assert abs(splitter.model_b_ratio - 0.2) < 1e-9

    def test_update_ratio_invalid_raises_error(self):
        splitter = TrafficSplitter()
        with pytest.raises(ValueError):
            splitter.update_ratio(2.0)

    def test_assign_variant_returns_valid_variant(self):
        splitter = TrafficSplitter()
        variant = splitter.assign_variant()
        assert variant in ("A", "B")

    def test_fifty_fifty_distribution(self):
        """Verify 50/50 split over 10,000 requests within 5% tolerance."""
        splitter = TrafficSplitter(model_a_ratio=0.5)
        counts = {"A": 0, "B": 0}

        for _ in range(10000):
            variant = splitter.assign_variant()
            counts[variant] += 1

        ratio_a = counts["A"] / 10000
        assert abs(ratio_a - 0.5) < 0.05, (
            f"Expected ~50% Model A, got {ratio_a:.1%}"
        )

    def test_seventy_thirty_distribution(self):
        """Verify 70/30 split over 10,000 requests within 5% tolerance."""
        splitter = TrafficSplitter(model_a_ratio=0.7)
        counts = {"A": 0, "B": 0}

        for _ in range(10000):
            variant = splitter.assign_variant()
            counts[variant] += 1

        ratio_a = counts["A"] / 10000
        assert abs(ratio_a - 0.7) < 0.05, (
            f"Expected ~70% Model A, got {ratio_a:.1%}"
        )

    def test_ninety_ten_distribution(self):
        """Verify 90/10 split over 10,000 requests within 5% tolerance."""
        splitter = TrafficSplitter(model_a_ratio=0.9)
        counts = {"A": 0, "B": 0}

        for _ in range(10000):
            variant = splitter.assign_variant()
            counts[variant] += 1

        ratio_a = counts["A"] / 10000
        assert abs(ratio_a - 0.9) < 0.05, (
            f"Expected ~90% Model A, got {ratio_a:.1%}"
        )

    def test_all_traffic_to_model_a(self):
        """When ratio is 1.0, all traffic goes to Model A."""
        splitter = TrafficSplitter(model_a_ratio=1.0)
        for _ in range(100):
            assert splitter.assign_variant() == "A"

    def test_all_traffic_to_model_b(self):
        """When ratio is 0.0, all traffic goes to Model B."""
        splitter = TrafficSplitter(model_a_ratio=0.0)
        for _ in range(100):
            assert splitter.assign_variant() == "B"

    def test_session_sticky_assignment(self):
        """Same session_id always gets the same variant."""
        splitter = TrafficSplitter(model_a_ratio=0.5)
        session_id = "user_42"
        first_variant = splitter.assign_variant(session_id=session_id)

        for _ in range(100):
            assert splitter.assign_variant(session_id=session_id) == first_variant

    def test_different_sessions_get_different_assignments(self):
        """Different session IDs can produce different assignments."""
        splitter = TrafficSplitter(model_a_ratio=0.5)
        variants = set()

        for i in range(100):
            variant = splitter.assign_variant(session_id=f"user_{i}")
            variants.add(variant)

        assert len(variants) == 2, "Expected both variants across different sessions"

    def test_ratio_update_affects_distribution(self):
        """Updating the ratio changes the traffic distribution."""
        splitter = TrafficSplitter(model_a_ratio=0.0)

        for _ in range(10):
            assert splitter.assign_variant() == "B"

        splitter.update_ratio(1.0)
        for _ in range(10):
            assert splitter.assign_variant() == "A"
