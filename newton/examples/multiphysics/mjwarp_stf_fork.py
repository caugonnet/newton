# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Intra-step fork of mujoco_warp's step (3.17.0) as cuda.stf tasks, with mujoco_warp and
Newton unchanged: a stand-in for the `mujoco_warp` module is set on one SolverMuJoCo
instance for the duration of a recording; its `step(m, d)` replays `mujoco_warp.step` /
`forward` stage by stage as tasks: the three chains read the scope token and may overlap,
the rest takes it read-write (newton._src.solvers.tasks).

Data flow (MuJoCo's forward pipeline), after kinematics + com_pos + camlight + flex + tendon:
  con  : [collision] -> make_constraint                (geometry, tendon lengths, qvel)
  mass : crb -> tendon_armature                        (cinert, ten_J)
  vel  : transmission -> fwd_velocity -> fwd_actuation (cdof, ten_J, qvel, ctrl)
then, sequential again: sensors/energy, fwd_acceleration(factorize) [M, bias, passive,
actuator], solve [efc, M], rne_postconstraint/sensor_acc, integrator.
`con` is recorded first: a stock Warp `capture_if`/`capture_while` inside a task re-attaches
to every graph leaf on resume (hecate 2026-11-07 finding), so the task that may hold one
(collision under m.opt.graph_conditional) must see no sibling leaves when it is recorded.

While installed, the solver instance declares `records_tasks`: its caller gives it a
scope, Newton's own code around the mujoco_warp call is recorded as tasks of that scope's
sequence (`installed` wraps `solver.step`), and the stand-in records the mujoco_warp step.
"""

import contextlib

from newton._src.solvers.tasks import scope_token, warp_task


class MjwForkShim:
    """Module stand-in; set as `solver._mujoco_warp` (instance attribute shadowing the class
    attribute) while recording, delete afterwards."""

    def __init__(self, mjw):
        self._mjw = mjw
        self.steps = 0
        self._outer = None  # the task open around Newton's own code in SolverMuJoCo.step

    def begin(self):
        self._outer = warp_task(scope_token().rw(), symbol="mujoco-newton")
        self._outer.__enter__()

    def end(self):
        outer, self._outer = self._outer, None
        outer.__exit__(None, None, None)

    def __getattr__(self, name):  # everything else Newton asks of mujoco_warp
        return getattr(self._mjw, name)

    def step(self, m, d):
        mjw = self._mjw
        from mujoco_warp._src import constraint, forward, sensor, smooth
        from mujoco_warp._src.types import DisableBit, EnableBit, IntegratorType

        sleep_enabled = bool(m.opt.enableflags & EnableBit.SLEEP) and not bool(m.opt.disableflags & DisableBit.ISLAND)
        assert not sleep_enabled, "fork written for sleep disabled"
        assert m.opt.integrator in (IntegratorType.EULER, IntegratorType.IMPLICITFAST, IntegratorType.IMPLICIT), m.opt.integrator
        assert not m.callback.control, "control callback: keep actuation in the join"
        forward.check_discrete(m)
        self.end()  # Newton's code before the mujoco_warp step
        seq = scope_token()
        with warp_task(seq.rw(), symbol="kinematics"):
            mjw.fwd_kinematics(m, d)
        with warp_task(seq.read(), symbol="con"):
            if m.opt.run_collision_detection:
                mjw.collision(m, d)
            constraint.make_constraint(m, d)
        with warp_task(seq.read(), symbol="mass"):
            smooth.crb(m, d)
            smooth.tendon_armature(m, d)
        with warp_task(seq.read(), symbol="vel"):
            smooth.transmission(m, d)
            mjw.fwd_velocity(m, d)
            mjw.fwd_actuation(m, d)
        with warp_task(seq.rw(), symbol="solve"):
            d.sensordata.zero_()
            sensor.sensor_pos(m, d)
            forward._energy_pos(m, d)
            sensor.sensor_vel(m, d)
            forward._energy_vel(m, d)
            mjw.fwd_acceleration(m, d, factorize=True)
            mjw.solve(m, d)
            if m.opt.run_rne_postconstraint or (not (m.opt.disableflags & DisableBit.SENSOR) and m.sensor_rne_postconstraint):
                smooth.rne_postconstraint(m, d)
            sensor.sensor_acc(m, d, skip_rne_postconstraint=True)
            if m.opt.integrator == IntegratorType.EULER:
                mjw.euler(m, d)
            else:
                mjw.implicit(m, d)
        self.steps += 1
        self.begin()  # Newton's code after it


@contextlib.contextmanager
def installed(solver, shim):
    assert not getattr(solver, "_use_speculative_contacts", False), "speculative contacts path not forked"
    step = solver.step

    def step_as_tasks(*args, **kwargs):
        shim.begin()
        try:
            return step(*args, **kwargs)
        finally:
            shim.end()

    solver._mujoco_warp = shim
    solver.records_tasks = True
    solver.step = step_as_tasks
    try:
        yield shim
    finally:
        del solver._mujoco_warp, solver.records_tasks, solver.step
