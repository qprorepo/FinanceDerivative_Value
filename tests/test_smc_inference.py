import numpy as np
import pytest

from qcmc.smc_inference import geometric_depth_schedule, run_particle_filter_bae


def test_geometric_depth_schedule_is_monotone_nondecreasing():
    sched = geometric_depth_schedule(n_iters=50, m_max=45, c=1.12)
    assert len(sched) == 50
    assert np.all(np.diff(sched) >= 0)
    assert sched.max() <= 45
    assert sched.min() >= 0
    assert sched[0] == 1  # floor(1.12**0) = 1


def test_geometric_depth_schedule_respects_cap():
    sched = geometric_depth_schedule(n_iters=200, m_max=10)
    assert sched.max() == 10


def test_particle_filter_theta_converges_toward_truth(rng):
    """With enough particles and iterations, the posterior mean of theta_A
    should land reasonably close to the true value (mirrors the
    manuscript's reported ~2% relative error after 100 iterations)."""
    theta_true, gamma_true, fidelity_true = 0.338, 1e-3, 0.999
    snapshots, posterior_mean, m_schedule = run_particle_filter_bae(
        theta_true,
        gamma_true,
        fidelity_true,
        n_iters=100,
        n_particles=3000,
        shots_per_depth=32,
        m_max=45,
        rng=np.random.default_rng(11),
    )
    rel_err = abs(posterior_mean["theta"] - theta_true) / theta_true
    assert rel_err < 0.10  # generous bound; manuscript reports ~2.2% on one seed
    assert set(snapshots.keys()) == {10, 50, 100}
    assert m_schedule[-1] <= 45


def test_particle_filter_snapshot_weights_are_normalised():
    snapshots, _, _ = run_particle_filter_bae(
        0.3,
        1e-3,
        0.999,
        n_iters=20,
        n_particles=500,
        shots_per_depth=16,
        m_max=20,
        rng=np.random.default_rng(2),
        record_every=(20,),
    )
    w = snapshots[20]["weights"]
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(w >= 0)


def test_particle_filter_fidelity_stays_within_prior_support():
    """Fidelity is physically bounded in [0, 1]; the filter must respect
    this even under resampling/rejuvenation jitter."""
    snapshots, posterior_mean, _ = run_particle_filter_bae(
        0.3,
        1e-3,
        0.999,
        n_iters=30,
        n_particles=500,
        shots_per_depth=16,
        m_max=20,
        rng=np.random.default_rng(3),
        record_every=(30,),
    )
    fid = snapshots[30]["fidelity"]
    assert np.all(fid <= 1.0)
    assert np.all(fid >= 0.95)  # prior lower bound
    assert 0.0 <= posterior_mean["fidelity"] <= 1.0
