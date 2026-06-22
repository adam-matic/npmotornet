"""Unit tests for MileusnicSpindle: verify additive vs multiplicative gamma behavior."""
import numpy as np
import pytest
from npmotornet.sensory import MileusnicSpindle, MuscleSpindle


DT = 0.001
BATCH = 1
N = 6


def make_length(val, n=N):
    return np.full((BATCH, n), val, dtype=np.float32)


def make_vel(val=0.0, n=N):
    return np.full((BATCH, n), val, dtype=np.float32)


def make_gamma(val, n=N):
    return np.full((BATCH, n), val, dtype=np.float32)


@pytest.fixture
def mileusnic():
    return MileusnicSpindle(n_receptors=N, optimal_length=0.08)


@pytest.fixture
def multiplicative():
    return MuscleSpindle(n_receptors=N, optimal_length=0.08)


# --- Baseline parity (gamma=0) ---

def test_baseline_no_gamma_Ia(mileusnic):
    L = make_length(0.08)
    v = make_vel(0.0)
    out = mileusnic.get_firing_rate(L, v, DT)
    np.testing.assert_allclose(out['Ia'], mileusnic.baseline_Ia, atol=1e-4)


def test_baseline_no_gamma_II(mileusnic):
    L = make_length(0.08)
    v = make_vel(0.0)
    out = mileusnic.get_firing_rate(L, v, DT)
    np.testing.assert_allclose(out['II'], mileusnic.baseline_II, atol=1e-4)


# --- Additive gamma: Ia rises with gamma_static regardless of L vs L0 ---

def test_additive_gamma_above_L0(mileusnic):
    """Gamma raises Ia when muscle is above L0."""
    L = make_length(0.085)  # above L0=0.08
    v = make_vel()
    out_no = mileusnic.get_firing_rate(L, v, DT, gamma_static=make_gamma(0.0))
    mileusnic.reset()
    out_gs = mileusnic.get_firing_rate(L, v, DT, gamma_static=make_gamma(1.0))
    assert np.all(out_gs['Ia'] > out_no['Ia']), "Additive gamma must raise Ia above L0"


def test_additive_gamma_at_L0(mileusnic):
    """Gamma raises Ia when muscle is exactly at L0."""
    L = make_length(0.08)  # exactly L0
    v = make_vel()
    out_no = mileusnic.get_firing_rate(L, v, DT, gamma_static=make_gamma(0.0))
    mileusnic.reset()
    out_gs = mileusnic.get_firing_rate(L, v, DT, gamma_static=make_gamma(1.0))
    assert np.all(out_gs['Ia'] > out_no['Ia']), "Additive gamma must raise Ia at L0"


def test_additive_gamma_below_L0(mileusnic):
    """Gamma raises Ia when muscle is below L0 (shortened — the critical test)."""
    L = make_length(0.075)  # below L0=0.08
    v = make_vel()
    out_no = mileusnic.get_firing_rate(L, v, DT, gamma_static=make_gamma(0.0))
    mileusnic.reset()
    out_gs = mileusnic.get_firing_rate(L, v, DT, gamma_static=make_gamma(1.0))
    assert np.all(out_gs['Ia'] > out_no['Ia']), "Additive gamma must raise Ia even below L0"


# --- Multiplicative gamma FAILS the below-L0 test ---

def test_multiplicative_gamma_fails_below_L0(multiplicative):
    """MuscleSpindle's multiplicative gamma does NOT raise Ia below L0."""
    L = make_length(0.075)
    v = make_vel()
    gamma_static = make_gamma(1.0)
    # MuscleSpindle needs reset between calls
    out_no = multiplicative.get_firing_rate(L, v, DT, gamma_static=np.zeros_like(L))
    multiplicative.reset()
    out_gs = multiplicative.get_firing_rate(L, v, DT, gamma_static=gamma_static)
    # Below L0, multiplicative gamma amplifies a negative length_deviation → lowers Ia
    # (or at best leaves it unchanged). Ia with gamma should NOT be consistently higher.
    Ia_no = out_no['Ia'].mean()
    Ia_gs = out_gs['Ia'].mean()
    assert Ia_gs <= Ia_no + 1.0, (
        f"Multiplicative gamma unexpectedly raised Ia below L0: {Ia_no:.1f} → {Ia_gs:.1f}"
    )


# --- Monotone Ia vs gamma_static sweep ---

def test_monotone_gamma_static_sweep(mileusnic):
    """Ia must be monotonically non-decreasing as gamma_static increases (0→1)."""
    L = make_length(0.075)  # below L0, worst case for multiplicative
    v = make_vel()
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    prev_Ia = -np.inf
    for g in gammas:
        mileusnic.reset()
        out = mileusnic.get_firing_rate(L, v, DT, gamma_static=make_gamma(g))
        curr_Ia = out['Ia'].mean()
        assert curr_Ia >= prev_Ia - 1e-3, f"Ia non-monotone at gamma={g}: {prev_Ia:.2f} → {curr_Ia:.2f}"
        prev_Ia = curr_Ia


# --- Dynamic gamma increases velocity sensitivity ---

def test_dynamic_gamma_velocity_sensitivity(mileusnic):
    """Higher gamma_dynamic should produce higher Ia for the same stretch velocity."""
    L = make_length(0.08)
    v = make_vel(0.1)
    out_no = mileusnic.get_firing_rate(L, v, DT, gamma_dynamic=make_gamma(0.0))
    mileusnic.reset()
    out_gd = mileusnic.get_firing_rate(L, v, DT, gamma_dynamic=make_gamma(1.0))
    assert np.all(out_gd['Ia'] > out_no['Ia']), "Dynamic gamma must increase velocity-driven Ia"


# --- Saturation clipping ---

def test_saturation_Ia(mileusnic):
    L = make_length(0.20)  # very long
    v = make_vel(1.0)
    out = mileusnic.get_firing_rate(L, v, DT, gamma_static=make_gamma(1.0))
    assert np.all(out['Ia'] <= mileusnic.saturation_Ia + 1e-4)


def test_no_negative_firing(mileusnic):
    L = make_length(0.01)  # very short
    v = make_vel(-1.0)
    out = mileusnic.get_firing_rate(L, v, DT)
    assert np.all(out['Ia'] >= 0.0)
    assert np.all(out['II'] >= 0.0)


# --- get_static_response convenience method ---

def test_static_response_matches_zero_velocity(mileusnic):
    L = make_length(0.082)
    gs = make_gamma(0.5)
    out_dynamic = mileusnic.get_firing_rate(L, make_vel(0.0), DT, gamma_static=gs)
    mileusnic.reset()
    out_static = mileusnic.get_static_response(L, gamma_static=gs)
    np.testing.assert_allclose(out_static['Ia'], out_dynamic['Ia'], atol=1e-3)
    np.testing.assert_allclose(out_static['II'], out_dynamic['II'], atol=1e-3)
