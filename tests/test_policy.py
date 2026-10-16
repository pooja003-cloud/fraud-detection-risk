"""Checks that src/policy.py reproduces the published Step 4 results.

Run from the repo root:  python -m pytest tests/   (or simply: python tests/test_policy.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policy import do_nothing, evaluate, recommended_policy, expected_cost_policy  # noqa: E402

ORDERS = pd.read_csv(ROOT / "app" / "test_scores.csv.gz")
A = json.loads((ROOT / "app" / "assumptions.json").read_text())
STEP4 = json.loads((ROOT / "reports" / "step4_results.json").read_text())["results_test"]


def _close(a, b, rel=1e-6):
    return abs(a - b) <= rel * max(1.0, abs(b))


def test_do_nothing_matches_step4():
    r = evaluate(ORDERS, do_nothing(ORDERS), A)
    assert _close(r["cost_per_30_days"], STEP4["0. Do nothing"]["cost_per_30_days"])


def test_recommended_policy_matches_step4():
    r = evaluate(ORDERS, recommended_policy(ORDERS, A, friction_penalty=4.0), A)
    expected = STEP4["D. Expected-cost within limits"]
    for key in ["cost_per_30_days", "fraud_ratio", "genuine_instant_approval_%", "reviews_per_day_max"]:
        assert _close(r[key], expected[key]), (key, r[key], expected[key])


def test_unconstrained_policy_matches_step4():
    r = evaluate(ORDERS, expected_cost_policy(ORDERS, A), A)
    assert _close(r["cost_per_30_days"], STEP4["C. Expected-cost policy"]["cost_per_30_days"])


def test_review_cap_is_respected():
    actions = recommended_policy(ORDERS, A)
    per_day = pd.Series(actions == 3).groupby(ORDERS["day"]).sum()
    assert per_day.max() <= A["review_capacity_per_day"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
