"""
NovaCart fraud-decision policy: cost model, policies and evaluation.

This is the reusable core of Step 4 (notebooks/04_cost_optimised_policy.ipynb), packaged so the
policy can be re-run outside the notebooks, e.g. by a dashboard or a scheduled job.

Inputs are pandas DataFrames with one row per order and the columns:
    p               calibrated fraud probability (0-1)
    TransactionAmt  order amount in $
    day             day index (used to enforce the daily review cap)
    isFraud         true label (only needed by `evaluate`)

Actions are encoded as integers: 0 approve, 1 step-up, 2 decline, 3 manual review.

Example
-------
>>> import json, pandas as pd
>>> from src.policy import recommended_policy, evaluate
>>> orders = pd.read_csv("app/test_scores.csv.gz")
>>> A = json.load(open("app/assumptions.json"))
>>> actions = recommended_policy(orders, A, friction_penalty=4.0)
>>> evaluate(orders, actions, A)["cost_per_30_days"]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACTIONS = ["approve", "step_up", "decline", "review"]
APPROVE, STEP_UP, DECLINE, REVIEW = range(4)

# Default cost assumptions (documented in docs/business_brief.md and Step 4).
DEFAULT_ASSUMPTIONS = {
    "chargeback_fee": 25.0,         # $ per chargeback (processor fees typically $15-$100)
    "vamp_fine": 8.0,               # $ per fraud transaction in a month above Visa's limit
    "vamp_limit": 0.015,            # Visa VAMP merchant threshold
    "target_ratio": 0.005,          # the brief's target
    "margin": 0.30,                 # gross margin lost on a lost sale
    "churn_prob": 0.30,             # chance a wrongly declined customer never returns
    "future_margin": 80.0,          # margin from that customer's future orders
    "sms_cost": 0.05,               # $ per step-up verification
    "stepup_abandon": 0.10,         # share of genuine customers who abandon at step-up
    "stepup_bypass": 0.10,          # share of fraudsters who pass step-up
    "review_cost": 5.0,             # analyst cost per reviewed order
    "analyst_miss": 0.10,           # share of reviewed frauds the analyst approves
    "analyst_wrong_decline": 0.05,  # share of reviewed genuine orders the analyst declines
    "review_capacity_per_day": 100,
}


def action_costs(p, amt, A, fraud_weight: float = 1.0, friction_penalty: float = 0.0) -> np.ndarray:
    """Expected $ cost of each action (columns = ACTIONS) for orders with fraud probability `p`.

    With `p` set to the true label (0/1) this gives the realised cost, so one function serves both
    deciding and scoring. `fraud_weight` and `friction_penalty` are policy dials only; reported costs
    always use the defaults (1.0 and 0.0).
    """
    p, amt = np.asarray(p, float), np.asarray(amt, float)
    fraud_loss = (amt + A["chargeback_fee"]) * fraud_weight
    lost_customer = amt * A["margin"] + A["churn_prob"] * A["future_margin"]
    lost_sale = amt * A["margin"]
    f = (1 - p) * friction_penalty
    return np.column_stack([
        p * fraud_loss,                                                                   # approve
        A["sms_cost"] + p * A["stepup_bypass"] * fraud_loss
        + (1 - p) * A["stepup_abandon"] * lost_sale + f,                                  # step-up
        (1 - p) * lost_customer + f,                                                      # decline
        A["review_cost"] + p * A["analyst_miss"] * fraud_loss
        + (1 - p) * A["analyst_wrong_decline"] * lost_customer + f,                       # review
    ])


def cap_reviews(candidate, priority, day, cap: int) -> np.ndarray:
    """Pick at most `cap` candidates per day, highest priority first. Returns a boolean mask."""
    d = pd.DataFrame({"day": np.asarray(day), "prio": np.where(candidate, priority, -np.inf),
                      "cand": np.asarray(candidate)})
    d = d.sort_values(["day", "prio"], ascending=[True, False])
    d["rank"] = d.groupby("day").cumcount()
    chosen = d["cand"] & (d["rank"] < cap)
    return chosen.reindex(range(len(d))).to_numpy()


def do_nothing(frame: pd.DataFrame) -> np.ndarray:
    """Policy 0: approve every order (NovaCart's current situation)."""
    return np.zeros(len(frame), dtype=int)


def tier_policy(frame: pd.DataFrame, t_low: float, t_high: float, A) -> np.ndarray:
    """Policies A/B: approve below t_low, step up below t_high, otherwise decline.
    Within the decline tier the least certain orders go to review, up to the daily cap."""
    p, day = frame["p"].to_numpy(), frame["day"].to_numpy()
    action = np.where(p < t_low, APPROVE, np.where(p < t_high, STEP_UP, DECLINE))
    reviewed = cap_reviews(action == DECLINE, -p, day, A["review_capacity_per_day"])
    action[reviewed] = REVIEW
    return action


def expected_cost_policy(frame: pd.DataFrame, A, fraud_weight: float = 1.0,
                         friction_penalty: float = 0.0) -> np.ndarray:
    """Policies C/D/E: each order gets its lowest expected-cost action (uses probability AND amount).
    Review slots go to the orders where an analyst saves the most money, up to the daily cap."""
    p, amt, day = frame["p"].to_numpy(), frame["TransactionAmt"].to_numpy(), frame["day"].to_numpy()
    C = action_costs(p, amt, A, fraud_weight, friction_penalty)
    auto_best = C[:, :3].argmin(axis=1)
    review_saving = C[:, :3].min(axis=1) - C[:, 3]
    reviewed = cap_reviews(review_saving > 0, review_saving, day, A["review_capacity_per_day"])
    return np.where(reviewed, REVIEW, auto_best)


def recommended_policy(frame: pd.DataFrame, A, friction_penalty: float = 4.0) -> np.ndarray:
    """Policy D, the recommended policy: expected-cost decisions within the brief's friction limit.
    The default penalty ($4) is the value chosen on the validation month in Step 4."""
    return expected_cost_policy(frame, A, friction_penalty=friction_penalty)


def evaluate(frame: pd.DataFrame, action, A, days: int | None = None) -> dict:
    """Score a set of decisions against the true labels: cost, fraud ratio, friction and review load."""
    y, amt = frame["isFraud"].to_numpy(), frame["TransactionAmt"].to_numpy()
    action = np.asarray(action)
    idx = np.arange(len(y))
    base_cost = action_costs(y, amt, A)[idx, action]

    bypass, aband = A["stepup_bypass"], A["stepup_abandon"]
    miss, wrong = A["analyst_miss"], A["analyst_wrong_decline"]
    chargebacks = np.choose(action, [y, y * bypass, 0 * y, y * miss])
    settled = np.choose(action, [np.ones_like(y, float), y * bypass + (1 - y) * (1 - aband),
                                 0.0 * y, y * miss + (1 - y) * (1 - wrong)])
    ratio = chargebacks.sum() / settled.sum()
    fines = chargebacks.sum() * A["vamp_fine"] if ratio > A["vamp_limit"] else 0.0

    genuine = y == 0
    lost_genuine = np.choose(action, [0 * y, (1 - y) * aband, (1 - y) * 1.0, (1 - y) * wrong])
    n_days = days or frame["day"].nunique()
    reviews_per_day = pd.Series(action == REVIEW).groupby(frame["day"].to_numpy()).sum()
    return {
        "total_cost": float(base_cost.sum() + fines),
        "fines": float(fines),
        "fraud_ratio": float(ratio),
        "fraud_$_stopped": float(1 - (amt * chargebacks).sum() / (amt * y).sum()),
        "approve_%": float((action == APPROVE).mean()), "step_up_%": float((action == STEP_UP).mean()),
        "decline_%": float((action == DECLINE).mean()), "review_%": float((action == REVIEW).mean()),
        "genuine_friction_%": float((action[genuine] != APPROVE).mean()),
        "genuine_instant_approval_%": float((action[genuine] == APPROVE).mean()),
        "genuine_lost_%": float(lost_genuine[genuine].mean()),
        "reviews_per_day_avg": float(reviews_per_day.mean()),
        "reviews_per_day_max": int(reviews_per_day.max()),
        "cost_per_30_days": float((base_cost.sum() + fines) / n_days * 30),
    }
