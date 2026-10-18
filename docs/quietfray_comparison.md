# QuietFray v0.1 — Mathematical Specification & Canonical Algorithm Comparison

**Document Version:** 1.0  
**Status:** Algorithmic Specification & Dependency Audit  
**Canonical Package:** `MendSeed/QuietFray`  
**Local Implementation:** `sdk/mendseed_ai/quietfray.py` (`QuietFrayEngine`)

---

## 1. Algorithmic Overview & Mathematical Formulation

QuietFray is a **deterministic, marked Hawkes point process model** designed to solve a foundational failure mode in public problem platforms: **the tyranny of raw net votes**.

On conventional feedback boards:
- Bot swarms and transient social-media outrage create rapid vote spikes ($H_0$), pushing trivial or coordinated complaints to the top.
- Chronic, life-affecting community suffering (such as rural cold storage failure or contaminated drinking water) accumulates reports quietly and steadily over weeks ($H_1$), but is buried by net vote counters ($U - D$) or naive recency decay.

QuietFray decomposes event arrivals into a **bivariate dual-timescale excitation process** with quality-weighted marks $q_j \in [0, 1]$:

### 1.1 Dual-Decay Branching Mass
Given event arrival times $t_j < t$ with marks $q_j$:
$$x_k(t) = \sum_{j: t_j < t} q_j \cdot \exp\left(-\frac{t - t_j}{\tau_k}\right), \quad k \in \{f, s\}$$

Where:
- **Fast Kernel ($\tau_f = 0.02\text{ days} = 28.8\text{ minutes}$):** Rapidly dissipates short-term coordination bursts and duplicate flurries.
- **Slow Kernel ($\tau_s = 3.0\text{ days}$):** Sustains genuine long-term community friction over multi-week observation windows.
- **Composite Score:**
  $$\text{Score}(t) = w_f \cdot x_f(t) + w_s \cdot x_s(t) \quad (w_f = 0.7, w_s = 0.3)$$

### 1.2 Event Mark Formulation
Marks $q_j$ scale the excitation of each event based on source credibility, description entropy, and burst damping:
$$q_j = \operatorname{clip}\left(c_j \cdot H_j \cdot (1 - s_j^{\text{burst}}), 0, 1\right)$$
- $c_j$: Model extraction confidence score $\in [0, 1]$.
- $H_j$: Normalized lexical entropy / information density $\in [0, 1]$.
- $s_j^{\text{burst}}$: Jaccard similarity to recent reports within the fast-decay window ($28.8$ minutes).

### 1.3 Streaming $O(1)$ State Recurrence
QuietFray operates in constant arithmetic time $O(1)$ without scanning historical rows:
1. **Decay states over silent interval $\Delta t = t_{\text{new}} - t_{\text{last}}$:**
   $$x_k(t^-) = x_k(t_{\text{last}}) \cdot \left(\operatorname{expm1}\left(-\frac{\Delta t}{\tau_k}\right) + 1.0\right)$$
2. **Subtract integrated intensity difference:**
   $$\Delta I = \sum_{k \in \{f, s\}} (\alpha_{1,k} - \alpha_{0,k}) \cdot x_k \cdot \tau_k \cdot \left[-\operatorname{expm1}\left(-\frac{\Delta t}{\tau_k}\right)\right]$$
3. **Update Log-Likelihood Ratio (LLR):**
   $$\text{LLR} \leftarrow \text{LLR} - \Delta I + \log \frac{\lambda_1(t^-)}{\lambda_0(t^-)}$$
4. **Increment state with new mark $q_{\text{new}}$:**
   $$x_k \leftarrow x_k(t^-) + q_{\text{new}}$$

---

## 2. Comparison: Research Prototype vs Canonical vs Local Implementation

| Dimension | Notebook Research Prototype | Canonical Specification (`MendSeed/QuietFray`) | Local Implementation (`sdk/mendseed_ai/quietfray.py`) |
|---|---|---|---|
| **Class Name** | `QuietFrayEngine` / `QuietFlayEngine` | `QuietFrayEngine` | `QuietFrayEngine` *(with `QuietFlayEngine` alias)* |
| **External Dependencies** | `numpy` (`np.expm1`, `np.log1p`) | None (Pure Python `math`) | Zero (Pure Python standard library `math`) |
| **Timescales** | Toy values: $\lambda_f = 0.5, \lambda_s = 0.05$ | $\tau_f = 0.02\text{ d}$, $\tau_s = 3.0\text{ d}$ | $\tau_f = 0.02\text{ d}$, $\tau_s = 3.0\text{ d}$ (Exact match) |
| **Weights** | $w_f = 0.7, w_s = 0.3$ | $w_f = 0.7, w_s = 0.3$ | $w_f = 0.7, w_s = 0.3$ (Exact match) |
| **Hawkes Parameters** | Not modeled | Calibrated $\alpha_{r0}, \alpha_{r1}, \alpha_{v0}, \alpha_{v1}$ | Calibrated $\alpha_{r0}, \alpha_{r1}, \alpha_{v0}, \alpha_{v1}$ (Exact match) |
| **Subcriticality Check** | Missing (explosive risk) | Mandatory: $\alpha_f \tau_f + \alpha_s \tau_s < 1.0$ | Mandatory: Enforced in `__init__` via `_validate_subcriticality` |
| **Streaming Recurrence** | Toy LLR ratio | Exact analytical integral $\Delta I$ | Exact analytical integral $\Delta I$ via `math.expm1` |
| **Numerical Stability** | Standard float64 | Guarded against division-by-zero & underflow | Guarded with `math.expm1` and sign-branching logit sigmoid |

---

## 3. Subcriticality Proof & Stability Guarantee

A multi-dimensional Hawkes process is subcritical (meaning the expected cluster size is finite and excitation cannot cascade to infinity) if and only if the spectral radius of the branching matrix $\Gamma$ satisfies $\rho(\Gamma) < 1.0$.

For each channel (Reports and Votes) under both hypotheses:
$$\nu = \alpha_f \cdot \tau_f + \alpha_s \cdot \tau_s$$

Evaluating with MendSeed parameters:
1. **Reports Under $H_0$ (Spam Burst):**
   $$\nu_{r0} = (15.0 \times 0.02) + (0.020 \times 3.0) = 0.300 + 0.060 = \mathbf{0.360} < 1.0 \quad \checkmark$$
2. **Reports Under $H_1$ (Persistent Friction):**
   $$\nu_{r1} = (0.50 \times 0.02) + (0.18 \times 3.0) = 0.010 + 0.540 = \mathbf{0.550} < 1.0 \quad \checkmark$$
3. **Votes Under $H_0$ (Bot Upvotes):**
   $$\nu_{v0} = (12.0 \times 0.02) + (0.025 \times 3.0) = 0.240 + 0.075 = \mathbf{0.315} < 1.0 \quad \checkmark$$
4. **Votes Under $H_1$ (Genuine Community Endorsement):**
   $$\nu_{v1} = (0.35 \times 0.02) + (0.16 \times 3.0) = 0.007 + 0.480 = \mathbf{0.487} < 1.0 \quad \checkmark$$

All four branching coefficients strictly satisfy $\nu < 1.0$. The process is mathematically guaranteed to be subcritical and non-explosive under all operational conditions.

---

## 4. Deterministic Test Fixtures

The local test suite (`tests/test_quietfray.py`) verifies the following invariants:
1. **Zero-Event Invariance:** An issue with zero events returns branching mass $= 0.0$.
2. **Exponential Half-Life Invariance:** An impulse of mark $1.0$ evaluated at $t = t_0 + \tau_f \ln 2$ decays to exactly $0.5$ in the fast channel.
3. **Sybil Dampening Invariance:** 100 arrivals within 15 minutes achieve lower composite score at $t = 14.1\text{ days}$ than 5 independent reports spaced over 14 days.
4. **Subcriticality Enforcement Invariance:** Passing parameters where $\alpha_f \tau_f + \alpha_s \tau_s \ge 1.0$ immediately raises `ValueError`.
