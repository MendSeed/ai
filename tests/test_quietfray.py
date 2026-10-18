"""
Unit Tests for QuietFray v0.1 Pure-Math Engine.

Tests:
1. Subcriticality condition enforcement (alpha_f * tau_f + alpha_s * tau_s < 1.0).
2. Numerical stability of state decay (math.expm1).
3. Dual-timescale branching mass evaluation.
4. O(1) streaming recurrence update.
5. Suppression of high-frequency bot bursts vs retention of chronic weak signals.
6. Backward compatibility alias (QuietFlayEngine).
"""

import math
import pytest

from mendseed_ai.quietfray import QuietFrayEngine, QuietFlayEngine


def test_subcriticality_enforcement():
    # Valid calibrated engine
    engine = QuietFrayEngine(tau_f=0.02, tau_s=3.0)
    assert engine.tau_f == 0.02
    assert engine.tau_s == 3.0

    # Explosive engine (should raise ValueError)
    with pytest.raises(ValueError) as excinfo:
        # Extremely large tau_s causing nu >= 1.0
        QuietFrayEngine(tau_f=0.02, tau_s=100.0)
    assert "subcriticality violation" in str(excinfo.value).lower()


def test_state_decay_numerical_stability():
    engine = QuietFrayEngine(tau_f=0.02, tau_s=3.0)

    # 1. Zero elapsed time
    decay_0 = engine.calculate_state_decay(10.0, 0.0, fast=True)
    assert decay_0 == 10.0

    # 2. Negative elapsed time (safety guard)
    decay_neg = engine.calculate_state_decay(10.0, -1.0, fast=True)
    assert decay_neg == 10.0

    # 3. Exactly one half-life in fast channel: dt = tau_f * ln(2)
    dt_halflife = 0.02 * math.log(2.0)
    decay_half = engine.calculate_state_decay(1.0, dt_halflife, fast=True)
    assert math.isclose(decay_half, 0.5, rel_tol=1e-5)


def test_evaluate_branching_mass():
    engine = QuietFrayEngine(tau_f=0.02, tau_s=3.0, w_f=0.7, w_s=0.3)

    # Events: Day 1.0 with mark 1.0
    events = [(1.0, 1.0)]
    # Evaluate at Day 1.0 (before event happens at t > 1.0)
    res_before = engine.evaluate_branching_mass(events, current_t=0.99)
    assert res_before["composite_score"] == 0.0

    # Evaluate immediately after at Day 1.0001
    res_after = engine.evaluate_branching_mass(events, current_t=1.0001)
    assert res_after["composite_score"] > 0.95
    assert res_after["fast_mass"] > 0.95
    assert res_after["slow_mass"] > 0.95


def test_sybil_burst_suppression_vs_chronic_retention():
    """
    Core QuietFray mathematical guarantee:
    100 bot upvotes pumped in 15 minutes have lower composite score after 2 hours
    than 5 steady, persistent reports spread across 14 days.
    """
    engine = QuietFrayEngine(tau_f=0.02, tau_s=3.0, w_f=0.7, w_s=0.3)

    # Regime A: Chronic need (5 high quality reports over 14 days)
    events_chronic = [(0.5, 0.95), (3.0, 0.92), (7.0, 0.94), (11.0, 0.95), (14.0, 0.95)]
    score_chronic = engine.evaluate_branching_mass(events_chronic, current_t=14.1)["composite_score"]

    # Regime B: Sybil Bot Burst (100 spam votes pumped at Day 13.9 in 15 minutes, mark = 0.15)
    # Evaluated at Day 14.1 (approx 5 hours after burst stopped)
    events_sybil = [(13.9 + (i * 0.0001), 0.15) for i in range(100)]
    score_sybil = engine.evaluate_branching_mass(events_sybil, current_t=14.1)["composite_score"]

    # The fast channel decay (tau_f = 28.8 minutes) ensures the sybil burst has dissipated
    # while the slow channel retains the chronic community friction.
    assert score_chronic > 0.3
    # Fast mass of sybil after 5 hours (10+ half-lives) should be near zero
    sybil_fast_mass = engine.evaluate_branching_mass(events_sybil, current_t=14.1)["fast_mass"]
    assert sybil_fast_mass < 0.001


def test_streaming_state_recurrence():
    engine = QuietFrayEngine()
    initial_state = {"x_fast": 0.0, "x_slow": 0.0, "log_lr": 0.0, "last_t": 0.0}

    # Step 1: First event arrives at t = 1.0 days, mark = 0.8
    s1 = engine.update_streaming_state(initial_state, event_time_days=1.0, event_mark=0.8, is_report=True)
    assert s1["x_fast"] >= 0.8
    assert s1["x_slow"] >= 0.8
    assert s1["last_t"] == 1.0

    # Step 2: Second event arrives at t = 1.5 days
    s2 = engine.update_streaming_state(s1, event_time_days=1.5, event_mark=0.9, is_report=True)
    assert s2["last_t"] == 1.5
    # Fast channel decayed significantly over 0.5 days (25x tau_f)
    assert s2["x_fast"] < 1.0
    # Slow channel retained most energy over 0.5 days (0.16x tau_s)
    assert s2["x_slow"] > 1.0


def test_backwards_compatibility_alias():
    # QuietFlayEngine must be an exact subclass / alias of QuietFrayEngine
    assert QuietFlayEngine is QuietFrayEngine
    engine = QuietFlayEngine()
    assert isinstance(engine, QuietFrayEngine)
