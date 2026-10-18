"""
QuietFray v0.1 — Pure-Python Temporal Evidence Ranking Engine for MendSeed.

A mathematically specified, incrementally updatable ranking model that separates
genuine, persistent community friction (H1) from transient spam bursts or legacy monopolies (H0).

Key properties:
- Zero external dependencies: uses only Python standard library `math` and `typing`.
- Numerically stable: uses math.expm1 and math.log1p to prevent floating-point underflow/overflow.
- O(1) streaming recurrence: updates in constant arithmetic operations per event without rescanning history.
- Provably subcritical: dual Hawkes kernel parameters strictly satisfy alpha_f * tau_f + alpha_s * tau_s < 1.

Canonical reference: MendSeed/QuietFray v0.1 specification.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union


class QuietFrayEngine:
    """
    QuietFray v0.1 Temporal Evidence Ranking Engine.
    Distinguishes short-term viral spikes from persistent unresolved friction.
    """

    # Calibrated Hawkes process parameters for dual-kernel excitation
    DEFAULT_PARAMETERS = {
        "tau_f": 0.02,   # Fast decay timescale: 0.02 days = 28.8 minutes
        "tau_s": 3.0,    # Slow decay timescale: 3.0 days
        "w_f": 0.7,      # Fast mass weight
        "w_s": 0.3,      # Slow mass weight
        "alpha_votes_h0": (12.0, 0.025),
        "alpha_votes_h1": (0.35, 0.16),
        "alpha_reports_h0": (15.0, 0.020),
        "alpha_reports_h1": (0.50, 0.18),
        "mu_default": 0.05,
    }

    def __init__(
        self,
        tau_f: float = 0.02,
        tau_s: float = 3.0,
        w_f: float = 0.7,
        w_s: float = 0.3,
        mu_rate: float = 0.05,
    ):
        """
        Initialize the QuietFray Engine.

        :param tau_f: Fast decay time constant in days (default: 0.02 days = 28.8 min)
        :param tau_s: Slow decay time constant in days (default: 3.0 days)
        :param w_f: Fast mass component weight (default: 0.7)
        :param w_s: Slow mass component weight (default: 0.3)
        :param mu_rate: Baseline Poisson arrival rate per day
        """
        self.tau_f = float(tau_f)
        self.tau_s = float(tau_s)
        self.lambda_f = 1.0 / self.tau_f
        self.lambda_s = 1.0 / self.tau_s
        self.w_f = float(w_f)
        self.w_s = float(w_s)
        self.mu_rate = float(mu_rate)

        # Hawkes kernel parameters for Reports channel
        self.alpha_r0_f = 15.0
        self.alpha_r0_s = 0.020
        self.alpha_r1_f = 0.50
        self.alpha_r1_s = 0.18

        # Hawkes kernel parameters for Votes channel
        self.alpha_v0_f = 12.0
        self.alpha_v0_s = 0.025
        self.alpha_v1_f = 0.35
        self.alpha_v1_s = 0.16

        self._validate_subcriticality()

    def _validate_subcriticality(self) -> None:
        """
        Enforce subcriticality condition:
        nu = alpha_f * tau_f + alpha_s * tau_s < 1.0
        Ensures bounded branching mass and finite expected cluster sizes.
        """
        nu_r0 = self.alpha_r0_f * self.tau_f + self.alpha_r0_s * self.tau_s
        nu_r1 = self.alpha_r1_f * self.tau_f + self.alpha_r1_s * self.tau_s
        nu_v0 = self.alpha_v0_f * self.tau_f + self.alpha_v0_s * self.tau_s
        nu_v1 = self.alpha_v1_f * self.tau_f + self.alpha_v1_s * self.tau_s

        for name, val in [("nu_r0", nu_r0), ("nu_r1", nu_r1), ("nu_v0", nu_v0), ("nu_v1", nu_v1)]:
            if val >= 1.0:
                raise ValueError(
                    f"Subcriticality violation in QuietFray parameter {name} = {val:.4f} >= 1.0. "
                    "Hawkes process would be explosive."
                )

    def calculate_state_decay(
        self, x_initial: float, delta_t_days: float, fast: bool = True
    ) -> float:
        """
        Decay an excitation state over delta_t days.
        Uses math.expm1 for numerically stable calculation when delta_t / tau is small:
        exp(-lambda * dt) = expm1(-lambda * dt) + 1.0
        """
        if delta_t_days <= 0.0:
            return float(x_initial)
        rate = self.lambda_f if fast else self.lambda_s
        decay_factor = math.expm1(-rate * delta_t_days) + 1.0
        return float(x_initial) * decay_factor

    @staticmethod
    def calculate_report_mark(
        confidence: float,
        diversity: float,
        recent_burst_similarity: float
    ) -> float:
        """
        Computes the candidate quality/excitation mark for a problem report:
        q_k = clip(c_k * H_k * (1 - s_k^burst), 0, 1)

        :param confidence: Model confidence in extraction [0, 1]
        :param diversity: Normalized description diversity [0, 1]
        :param recent_burst_similarity: Semantic similarity to reports in recent burst window [0, 1]
        :return: Bounded mark in [0.0, 1.0]
        """
        raw = confidence * diversity * (1.0 - recent_burst_similarity)
        return max(0.0, min(1.0, float(raw)))

    def evaluate_branching_mass(
        self,
        events: Sequence[Tuple[float, float]],
        current_t: float
    ) -> Dict[str, float]:
        """
        Computes:
        x_{m,k}(t) = sum_{t_j < t} q_j * exp(-(t - t_j) / tau_k)
        where each event is a tuple (timestamp_in_days, quality_weight_q_j).

        :param events: Sequence of (timestamp_days, mark_q)
        :param current_t: Current evaluation timestamp in days
        :return: Dict containing fast_mass, slow_mass, and composite_score
        """
        fast_mass = 0.0
        slow_mass = 0.0

        for t_j, q_j in events:
            if t_j < current_t:
                dt = current_t - t_j
                fast_mass += self.calculate_state_decay(q_j, dt, fast=True)
                slow_mass += self.calculate_state_decay(q_j, dt, fast=False)

        composite_score = (self.w_f * fast_mass) + (self.w_s * slow_mass)
        return {
            "fast_mass": round(fast_mass, 6),
            "slow_mass": round(slow_mass, 6),
            "composite_score": round(composite_score, 6),
        }

    def compute_integrated_intensity_diff(
        self,
        x_fast: float,
        x_slow: float,
        delta_t_days: float,
        is_report: bool = True
    ) -> float:
        """
        Computes the exact analytical integral of the intensity difference
        between H1 and H0 over interval delta_t:
        Delta_I = sum_{k in {f,s}} (alpha_1,k - alpha_0,k) * x_k * tau_k * [-expm1(-dt / tau_k)]
        """
        if delta_t_days <= 0.0:
            return 0.0

        if is_report:
            a0_f, a0_s = self.alpha_r0_f, self.alpha_r0_s
            a1_f, a1_s = self.alpha_r1_f, self.alpha_r1_s
        else:
            a0_f, a0_s = self.alpha_v0_f, self.alpha_v0_s
            a1_f, a1_s = self.alpha_v1_f, self.alpha_v1_s

        diff_f = (a1_f - a0_f) * x_fast * self.tau_f * (-math.expm1(-self.lambda_f * delta_t_days))
        diff_s = (a1_s - a0_s) * x_slow * self.tau_s * (-math.expm1(-self.lambda_s * delta_t_days))

        return diff_f + diff_s

    @staticmethod
    def compute_log_likelihood_ratio(base_evidence: float, test_evidence: float) -> float:
        """
        Numerically stable log-likelihood ratio calculation using math.log1p:
        LLR = log(test / base)
        Protects against zero division and floating-point cancellation.
        """
        ratio = (test_evidence + 1e-9) / (base_evidence + 1e-9)
        if ratio >= 1.0:
            return math.log1p(ratio - 1.0)
        return -math.log1p((1.0 / ratio) - 1.0)

    @staticmethod
    def logit_to_probability(log_odds: float) -> float:
        """
        Numerically stable sigmoid conversion from log-odds to calibrated probability.
        Uses sign-dependent branch to prevent overflow: sigma(x) = 1 / (1 + exp(-|x|))
        """
        if log_odds >= 0.0:
            z = math.exp(-log_odds)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(log_odds)
            return z / (1.0 + z)

    def update_streaming_state(
        self,
        current_state: Dict[str, float],
        event_time_days: float,
        event_mark: float,
        is_report: bool = True
    ) -> Dict[str, float]:
        """
        O(1) Streaming State Recurrence for an incoming event.
        Updates state in constant arithmetic time without scanning history.

        :param current_state: State dictionary holding x_fast, x_slow, log_lr, last_t
        :param event_time_days: Timestamp of new event
        :param event_mark: Computed event mark q_j
        :param is_report: True if problem report, False if vote
        :return: Updated state dictionary
        """
        last_t = current_state.get("last_t", event_time_days)
        delta_t = max(0.0, event_time_days - last_t)

        # 1. Decay previous states
        x_f = self.calculate_state_decay(current_state.get("x_fast", 0.0), delta_t, fast=True)
        x_s = self.calculate_state_decay(current_state.get("x_slow", 0.0), delta_t, fast=False)

        # 2. Subtract integrated intensity over silent interval
        int_diff = self.compute_integrated_intensity_diff(x_f, x_s, delta_t, is_report=is_report)
        log_lr = current_state.get("log_lr", 0.0) - int_diff

        # 3. Calculate instantaneous intensity at t^-
        if is_report:
            lam_0 = self.mu_rate + self.alpha_r0_f * x_f + self.alpha_r0_s * x_s
            lam_1 = self.mu_rate + self.alpha_r1_f * x_f + self.alpha_r1_s * x_s
        else:
            lam_0 = self.mu_rate + self.alpha_v0_f * x_f + self.alpha_v0_s * x_s
            lam_1 = self.mu_rate + self.alpha_v1_f * x_f + self.alpha_v1_s * x_s

        # 4. Add event's log-intensity increment
        event_lr = self.compute_log_likelihood_ratio(lam_0, lam_1)
        log_lr += event_lr

        # 5. Increment state with new event mark
        x_f += event_mark
        x_s += event_mark

        return {
            "x_fast": round(x_f, 8),
            "x_slow": round(x_s, 8),
            "log_lr": round(log_lr, 8),
            "last_t": event_time_days,
            "estimated_p_h1": round(self.logit_to_probability(log_lr), 6)
        }


# Backwards compatibility alias for the legacy typo
QuietFlayEngine = QuietFrayEngine
