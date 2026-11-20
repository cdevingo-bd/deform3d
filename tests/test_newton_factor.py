"""Newton Cholesky policy for unpinned MuJoCo Warp builds (a patched 3.17.0 source build)."""

from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

from deform3d_warp.mujoco_newton_factor import NewtonFactorPolicy, hook_is_reviewed

pytestmark = pytest.mark.integration
REVIEWED, WHY = hook_is_reviewed()
cuda = pytest.mark.skipif(not wp.is_cuda_available(), reason="requires CUDA")
reviewed = pytest.mark.skipif(not REVIEWED, reason=f"Newton factor hook not reviewed here: {WHY}")

# The repo's bag example: Newton solver, nv=357 (above the small-system path), and live contacts
# between the shell, the grid package, the fruit and the pads. It is stable on MJWarp 3.17.
SCENE = (Path(__file__).resolve().parents[1] / "examples" / "mujoco_bag.xml").read_text()


def sparse_scene(nbox=6, nsphere=30):
    """Boxes tumbling onto a plane plus hovering contact-free spheres.

    Only the boxes ever couple in the Newton Hessian; every sphere DOF has no off-diagonal entry.
    That is the structure of a flex or free-body scene, where the coupled-block solve applies.
    """
    boxes = "".join(
        f'<body name="box{i}" pos="{0.4 * (i % 3) - 0.4} {0.4 * (i // 3) - 0.2} {0.18 + 0.02 * i}" '
        f'euler="{12 * i} {8 * (i % 2)} 5"><freejoint/><geom type="box" size=".06 .05 .04" mass=".5"/></body>'
        for i in range(nbox))
    spheres = "".join(
        f'<body name="s{i}" pos="{0.15 * (i % 6) - 0.4} {0.15 * (i // 6) - 0.4} 2" gravcomp="1"><freejoint/>'
        f'<geom type="sphere" size=".03" mass=".1" contype="0" conaffinity="0"/></body>'
        for i in range(nsphere))
    return (f'<mujoco><option timestep="0.002" integrator="Euler" solver="Newton" gravity="0 0 -9.81"/>'
            f'<worldbody><geom type="plane" size="3 3 .1"/>{boxes}{spheres}</worldbody></mujoco>')


SPARSE = sparse_scene()


def build(nworld=2, xml=SCENE, warm=300):
    cm = mujoco.MjModel.from_xml_string(xml)
    cd = mujoco.MjData(cm)
    mujoco.mj_forward(cm, cd)
    for _ in range(warm):  # reach a state with live contacts
        mujoco.mj_step(cm, cd)
    model = mjw.put_model(cm)
    return cm, cd, model, lambda: mjw.put_data(cm, cd, nworld=nworld, nconmax=64 * nworld, njmax=512)


def capture(fn):
    fn()
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        fn()
    wp.synchronize()
    return cap.graph


CASES = {
    # id: (scene, warm CPU steps, policy kwargs, what must be true of the coupled sets)
    "dense_method_on_bag": (SCENE, 300, {"method": "dense"}, None),
    "coupled_on_sparse": (SPARSE, 40, {}, "solved_here"),
    "coupled_overflowing_partly": (SPARSE, 40, {"max_coupled": 32}, "mixed"),
    "coupled_on_dense_bag": (SCENE, 300, {}, "all_fallback"),
    # Structural marking, with the verification scan on: status bit 4 would flag a missed coupling.
    "jacobian_structure_on_sparse": (SPARSE, 40, {"structure": "jacobian", "verify_structure": True}, "solved_here"),
    # Same, but with a limit low enough that every contact solve is rescanned for its exact set.
    "jacobian_structure_forced_rescan": (SPARSE, 40, {"structure": "jacobian", "verify_structure": True,
                                                      "rescan_above": 4}, "rescanned"),
}


@cuda
@reviewed
@pytest.mark.parametrize("case", list(CASES))
def test_policy_is_used_and_matches_native_within_native_noise(case):
    xml, warm, kwargs, expectation = CASES[case]
    with wp.ScopedDevice("cuda:0"):
        cm, cpu, model, fresh = build(xml=xml, warm=warm)
        policy = NewtonFactorPolicy(model, 2, **kwargs)
        native, control, patched = fresh(), fresh(), fresh()
        g_native = capture(lambda: mjw.step(model, native))
        g_control = capture(lambda: mjw.step(model, control))
        from mujoco_warp._src import solver

        before = solver._cholesky_factorize_solve
        with policy.capture_policy():
            assert solver._cholesky_factorize_solve is not before
            g_patched = capture(lambda: mjw.step(model, patched))
        assert solver._cholesky_factorize_solve is before, "hook must be restored on exit"
        for d in (native, control, patched):
            for name in ("qpos", "qvel"):
                getattr(d, name).assign(np.tile(getattr(cpu, name), (2, 1)).astype(np.float32))
            d.time.zero_()
        policy.status.zero_()
        if policy.fallbacks is not None:
            policy.fallbacks.zero_()
        for _ in range(100):
            for g in (g_native, g_control, g_patched):
                wp.capture_launch(g)
        wp.synchronize()
        assert int(patched.nefc.numpy().max()) > 0, "scene must produce constraints"
        # Evidence the replacement actually ran, not just that the answer is right.
        np.testing.assert_array_equal(policy.status.numpy(), 0)
        if expectation is not None:
            assert 0 < int(policy.coupled.numpy().max()) <= cm.nv
            fallbacks = policy.fallbacks.numpy()
            if expectation == "solved_here":
                assert fallbacks.sum() == 0 and int(policy.coupled.numpy().max()) < cm.nv // 2
            elif expectation == "rescanned":
                assert fallbacks.sum() == 0 and int(policy.rescans.numpy().max()) > 0
            elif expectation == "mixed":
                assert fallbacks.sum() > 0, "max_coupled=32 is below the contact-time coupled size"
            elif expectation == "all_fallback":
                assert int(policy.coupled.numpy().min()) == cm.nv, "the discrete flex Hessian is dense"
                assert fallbacks.min() > 0
        else:
            assert (policy.workspace.initialized.numpy() == 1).all()
        qn, qc, qp = native.qpos.numpy(), control.qpos.numpy(), patched.qpos.numpy()
        assert np.isfinite(qp).all()
        noise = max(float(np.abs(qc - qn).max()), 1e-6)
        # MJWarp is not bitwise repeatable here, so judge against the native-vs-native control.
        assert float(np.abs(qp - qn).max()) <= 10 * noise


@cuda
@reviewed
def test_other_models_pass_through_to_the_native_hook():
    with wp.ScopedDevice("cuda:0"):
        _, _, model, fresh = build()
        _, _, other, fresh_other = build()
        policy = NewtonFactorPolicy(model, 2)
        d = fresh_other()
        with policy.capture_policy():
            mjw.step(other, d)  # a different Model object must not touch our workspace
        wp.synchronize()
        assert (policy.workspace.initialized.numpy() == 0).all()


@cuda
@reviewed
def test_rejects_unsupported_configurations():
    with wp.ScopedDevice("cuda:0"):
        assert 'solver="Newton"' in SCENE
        cg = SCENE.replace('solver="Newton"', 'solver="CG"')
        _, _, model, _ = build(xml=cg)
        with pytest.raises(ValueError, match="Newton"):
            NewtonFactorPolicy(model, 2)
        _, _, newton, _ = build()
        with pytest.raises(ValueError, match="nworld"):
            NewtonFactorPolicy(newton, 0)
        with pytest.raises(ValueError, match="method"):
            NewtonFactorPolicy(newton, 2, method="sparse")
        with pytest.raises(ValueError, match="structure"):
            NewtonFactorPolicy(newton, 2, structure="guess")
        with pytest.raises(ValueError, match="verify_structure"):
            NewtonFactorPolicy(newton, 2, verify_structure=True)
        with pytest.raises(ValueError, match="method='coupled'"):
            NewtonFactorPolicy(newton, 2, method="dense", structure="jacobian")
        # structural marking models how MJWarp assembles H, so it refuses what it does not model
        _, _, discrete, _ = build()
        with pytest.raises(ValueError, match="not supported"):
            NewtonFactorPolicy(discrete, 2, structure="jacobian")
        _, _, elliptic, _ = build(xml=SPARSE.replace("<option ", '<option cone="elliptic" '), warm=1)
        with pytest.raises(ValueError, match="elliptic"):
            NewtonFactorPolicy(elliptic, 2, structure="jacobian")


@cuda
@reviewed
def test_structural_marking_refuses_an_unreviewed_hessian_assembly(monkeypatch):
    from deform3d_warp import mujoco_newton_factor as factor

    ok, _ = factor.assembly_is_reviewed()
    assert ok, "the reviewed build's Hessian assembly must match its recorded hash"
    monkeypatch.setattr(factor, "REVIEWED_ASSEMBLIES", {})
    with wp.ScopedDevice("cuda:0"):
        _, _, model, _ = build(xml=SPARSE, warm=1)
        with pytest.raises(RuntimeError, match="assembly not reviewed"):
            NewtonFactorPolicy(model, 2, structure="jacobian")
        NewtonFactorPolicy(model, 2)  # the scan needs no assumption about assembly


@cuda
@pytest.mark.skipif(REVIEWED, reason="only meaningful where the hook is not reviewed")
def test_refuses_to_install_over_an_unreviewed_hook():
    with wp.ScopedDevice("cuda:0"):
        _, _, model, _ = build()
        with pytest.raises(RuntimeError, match="not reviewed"):
            NewtonFactorPolicy(model, 2)


class _Recorder:
    """Stands in for the workspace so dispatch arguments can be inspected without running a solve."""

    def __init__(self):
        self.calls = []

    def enqueue(self, *args):
        self.calls.append(args)


def _fake(nworld, device, nefc):
    from types import SimpleNamespace

    d = SimpleNamespace(nworld=nworld, nefc=nefc, qpos=SimpleNamespace(device=device))
    ctx = SimpleNamespace(quad_changed_count="QUAD", state_changed_count="STATE")
    return d, ctx


@cuda
@reviewed
def test_dispatch_passes_each_count_to_the_right_slot():
    from mujoco_warp._src import solver

    with wp.ScopedDevice("cuda:0"):
        _, _, model, _ = build()
        policy = NewtonFactorPolicy(model, 2)
        policy.workspace = _Recorder()
        d, ctx = _fake(2, policy.device, "NEFC")
        with policy.capture_policy():
            hook = solver._cholesky_factorize_solve
            hook(model, d, ctx, False, False)
            hook(model, d, ctx, True, False)
            hook(model, d, ctx, False, True)
            hook(model, d, ctx, True, True)
        # enqueue(ctx, quad_count, state_count, skip_unchanged, skip_noflip)
        got = [(c[1], c[2], c[3], c[4]) for c in policy.workspace.calls]
        assert got == [
            ("NEFC", "NEFC", False, False),
            ("QUAD", "NEFC", True, False),
            ("NEFC", "STATE", False, True),
            ("QUAD", "STATE", True, True),
        ]


@cuda
@reviewed
def test_dispatch_rejects_mismatched_world_count_or_device():
    from mujoco_warp._src import solver

    with wp.ScopedDevice("cuda:0"):
        _, _, model, _ = build()
        policy = NewtonFactorPolicy(model, 2)
        policy.workspace = _Recorder()
        with policy.capture_policy():
            hook = solver._cholesky_factorize_solve
            d, ctx = _fake(3, policy.device, "NEFC")
            with pytest.raises(ValueError, match="mismatch"):
                hook(model, d, ctx)
            d, ctx = _fake(2, wp.get_device("cpu"), "NEFC")
            with pytest.raises(ValueError, match="mismatch"):
                hook(model, d, ctx)
        assert policy.workspace.calls == []
