"""Dense effective-metric solve: gating, agreement with native MJWarp, and fallback."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

from deform3d_warp.mujoco_effective_banded import (BandedEffectiveSolve, plan,
                                                   reverse_cuthill_mckee, worthwhile)
from deform3d_warp.mujoco_effective_dense import DenseEffectiveSolve, supported

pytestmark = pytest.mark.integration
cuda = pytest.mark.skipif(not wp.is_cuda_available(), reason="dense effective solve requires CUDA")

FLEX = """<mujoco><option timestep="0.002" integrator="discrete" tolerance="1e-8" iterations="400"/>
  <worldbody><flexcomp name="blob" type="grid" dim="3" count="3 3 3" spacing=".05 .05 .05"
    pos="0 0 .5" mass=".2" radius=".002">
    <contact contype="0" conaffinity="0"/>
    <elasticity young="5e3" poisson=".3" damping=".02"/></flexcomp>
    <body pos="1 0 .5"><joint type="slide" axis="1 0 0" stiffness="800" springref=".1" damping="20"/>
      <geom type="sphere" size=".05" mass=".5" contype="0" conaffinity="0"/></body>
  </worldbody></mujoco>"""


def build(xml, nworld=2):
    cm = mujoco.MjModel.from_xml_string(xml)
    cd = mujoco.MjData(cm)
    mujoco.mj_forward(cm, cd)
    model = mjw.put_model(cm)
    data = mjw.put_data(cm, cd, nworld=nworld, nconmax=32, njmax=64)
    return cm, model, data


@cuda
@pytest.mark.parametrize("solver_class", [DenseEffectiveSolve, BandedEffectiveSolve])
def test_direct_solve_matches_native_operator(solver_class):
    from mujoco_warp._src import forward

    with wp.ScopedDevice("cuda:0"):
        _, model, data = build(FLEX)
        ok, reason = supported(model)
        assert ok, reason
        dense = solver_class(model, data)
        for _ in range(3):
            mjw.step(model, data)
        forward.fwd_position(model, data, factorize=False)
        forward.fwd_velocity(model, data)
        forward.fwd_actuation(model, data)
        data.qacc_smooth.zero_()
        forward.fwd_acceleration(model, data, factorize=True)
        native = data.qacc_smooth.numpy().astype(np.float64)
        data.qacc_smooth.zero_()
        with dense.capture_policy():
            forward.fwd_acceleration(model, data, factorize=True)
        got = data.qacc_smooth.numpy().astype(np.float64)
        # Independent check: residual of the NATIVE operator M_hat (eff_mul_m), not of our own
        # assembled matrix, so a wrong assembly cannot agree with itself.
        from mujoco_warp._src import derivative

        applied = wp.zeros_like(data.qacc_smooth)
        derivative.eff_mul_m(model, data, applied, data.qacc_smooth)
        rhs = (data.qfrc_smooth.numpy() + data.efm_c.numpy() + data.efm_ca.numpy()).astype(np.float64)
        # Group-wise relative error (flex dofs, then the joint dof last): a single global scale
        # would hide a wrong term on whichever group has the smaller forces.
        applied_np = applied.numpy()

        def group_error(sl):
            floor = 1e-3 * np.abs(rhs[:, sl]).max()
            return (np.abs(applied_np[:, sl] - rhs[:, sl]) / np.maximum(np.abs(rhs[:, sl]), floor)).max()

        residual = max(group_error(slice(0, -1)), group_error(slice(-1, None)))
        np.testing.assert_array_equal(dense.status.numpy(), 0)
        # float32 evaluation of the native operator sets a ~1e-4..1e-3 floor on small dofs; the
        # terms this guards (K, diagonal, mass) each move the result by >= 1e-2.
        assert residual < 2e-3, residual
        # The scene must exercise all three assembled terms, or mutations of one go unnoticed.
        assert np.abs(data.efm_diag.numpy()).max() > 0 and np.abs(data.efm_K_val.numpy()).max() > 0
        assert np.abs(got - native).max() / np.abs(native).max() < 1e-3  # both are float32 solves


@cuda
@pytest.mark.parametrize("solver_class", [DenseEffectiveSolve, BandedEffectiveSolve])
def test_unsupported_models_are_rejected_not_silently_wrong(solver_class):
    rigid = "<mujoco><option integrator='discrete'/><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>"
    with wp.ScopedDevice("cuda:0"):
        _, model, data = build(rigid, nworld=1)
        ok, reason = supported(model)
        assert not ok and "flex" in reason
        with pytest.raises(ValueError, match="unsupported"):
            solver_class(model, data)


def test_reverse_cuthill_mckee_is_a_permutation_that_shrinks_bandwidth():
    n = 60
    rng = np.random.default_rng(3)
    shuffle = rng.permutation(n)  # a path graph whose labels are scrambled
    adjacency = [set() for _ in range(n)]
    for a, b in zip(shuffle[:-1], shuffle[1:]):
        adjacency[a].add(int(b))
        adjacency[b].add(int(a))
    order = reverse_cuthill_mckee(adjacency)
    assert sorted(order.tolist()) == list(range(n))
    position = np.empty(n, dtype=int)
    position[order] = np.arange(n)
    width = max(abs(position[a] - position[b]) for a in range(n) for b in adjacency[a])
    scrambled = max(abs(a - b) for a in range(n) for b in adjacency[a])
    assert width == 1 and scrambled > 10


@cuda
def test_banded_plan_is_narrower_than_dense_for_a_flex_blob():
    with wp.ScopedDevice("cuda:0"):
        _, model, _ = build(FLEX)
        _, inverse, band = plan(model, 16)
        assert sorted(inverse.tolist()) == list(range(model.nv))
        blocks = (model.nv + 15) // 16
        assert 0 <= band < blocks
        assert worthwhile(model, 16) == (2 * band + 1 < blocks)


@cuda
@pytest.mark.parametrize("solver_class", [DenseEffectiveSolve, BandedEffectiveSolve])
def test_residual_gate_flags_an_ill_conditioned_system(solver_class):
    from mujoco_warp._src import forward

    # young=5e6 sat right at the 1e-4 threshold (the banded gate missed one world in ~1 of 12
    # runs on 4.0.1); 5e7 is flagged in every trial on both hosts, and 5e8 goes non-finite instead.
    stiff = FLEX.replace('young="5e3" poisson=".3" damping=".02"', 'young="5e7" poisson=".3" damping="2"')
    with wp.ScopedDevice("cuda:0"):
        _, model, data = build(stiff)
        solver = solver_class(model, data)
        for _ in range(3):
            mjw.step(model, data)
        forward.fwd_position(model, data, factorize=False)
        forward.fwd_velocity(model, data)
        forward.fwd_actuation(model, data)
        with solver.capture_policy():
            forward.fwd_acceleration(model, data, factorize=True)
        assert (solver.status.numpy() & 2).all(), "float32 residual gate must report this system"
