"""Simulate traffic to the A/B testing API.

Sends configurable numbers of requests with realistic feature vectors
sampled from the Breast Cancer dataset and reports summary statistics.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"


def load_sample_features() -> list[float]:
    """Load sample input features from the metadata file."""
    sample_path = "api/models/sample_input.json"
    if os.path.exists(sample_path):
        with open(sample_path, "r") as f:
            data = json.load(f)
            return data["sample_features"]
    return [float(x) for x in np.random.randn(30)]


def generate_features(base_features: list[float]) -> list[float]:
    """Generate a slightly perturbed version of the base features.

    Adds small Gaussian noise to simulate realistic variance in incoming
    data while keeping features within a plausible range.
    """
    noise = np.random.normal(0, 0.1, len(base_features))
    return [round(float(f + n), 4) for f, n in zip(base_features, noise)]


def check_health() -> bool:
    """Verify the API is healthy before sending traffic."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  API status: {data['status']}")
            print(f"  Models loaded: {data['models_loaded']}")
            return True
    except requests.ConnectionError:
        pass
    return False


def simulate_traffic(num_requests: int, delay_ms: int = 10) -> None:
    """Send simulated prediction requests to the API.

    Args:
        num_requests: Number of requests to send.
        delay_ms: Delay between requests in milliseconds.
    """
    print(f"\nSimulating {num_requests} requests to {PREDICT_ENDPOINT}")
    print(f"  Delay between requests: {delay_ms}ms")
    print("-" * 50)

    base_features = load_sample_features()
    variant_counts = {"A": 0, "B": 0}
    latencies = []
    errors = 0

    for i in range(num_requests):
        features = generate_features(base_features)
        payload = {"features": features}

        if random.random() < 0.3:
            payload["session_id"] = f"user_{random.randint(1, 50)}"

        try:
            start = time.perf_counter()
            resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=10)
            elapsed = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                variant_counts[data["model_variant"]] += 1
                latencies.append(elapsed)
            else:
                errors += 1

        except requests.RequestException:
            errors += 1

        if (i + 1) % 100 == 0 or (i + 1) == num_requests:
            print(f"  Progress: {i + 1}/{num_requests} requests sent")

        time.sleep(delay_ms / 1000)

    total = variant_counts["A"] + variant_counts["B"]
    print("\n" + "=" * 50)
    print("Traffic Simulation Summary")
    print("=" * 50)
    print(f"  Total requests sent:    {num_requests}")
    print(f"  Successful responses:   {total}")
    print(f"  Errors:                 {errors}")
    print(f"  Model A requests:       {variant_counts['A']} "
          f"({variant_counts['A'] / total * 100:.1f}%)" if total > 0 else "")
    print(f"  Model B requests:       {variant_counts['B']} "
          f"({variant_counts['B'] / total * 100:.1f}%)" if total > 0 else "")
    if latencies:
        print(f"  Avg round-trip latency: {np.mean(latencies):.1f} ms")
        print(f"  P95 round-trip latency: {np.percentile(latencies, 95):.1f} ms")
    print("=" * 50)


def main() -> None:
    """Entry point for the traffic simulation script."""
    parser = argparse.ArgumentParser(
        description="Simulate traffic to the A/B testing API"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=500,
        help="Number of requests to send (default: 500)",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=10,
        help="Delay between requests in ms (default: 10)",
    )
    args = parser.parse_args()

    print("Checking API health...")
    if not check_health():
        print("ERROR: API is not reachable. Is the service running?", file=sys.stderr)
        sys.exit(1)

    simulate_traffic(args.num_requests, args.delay_ms)


if __name__ == "__main__":
    main()
