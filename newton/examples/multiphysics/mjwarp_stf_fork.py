# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Intra-step fork of mujoco_warp's step (3.17.0) as cuda.stf tasks, with mujoco_warp and
Newton unchanged: a stand-in for the `mujoco_warp` module is set on one SolverMuJoCo
instance for the duration of a recording; its `step(m, d)` replays `mujoco_warp.step` /
`forward` stage by stage and moves the task boundary as it goes.

Data flow (MuJoCo's forward pipeline), after kinematics + com_pos + camlight + flex + tendon:
  con  : [collision] -> make_constraint                (geometry, tendon lengths, qvel)
  mass : crb -> tendon_armature                        (cinert, ten_J)
  vel  : transmission -> fwd_velocity -> fwd_actuation (cdof, ten_J, qvel, ctrl)
  join : sensors/energy, fwd_acceleration(factorize) [M, bias, passive, actuator],
         solve [efc, M], rne_postconstraint/sensor_acc, integrator.
`con` is recorded first: a stock Warp `capture_if`/`capture_while` inside a task re-attaches
to every graph leaf on resume (hecate 2026-11-07 finding), so the branch that may hold one
(collision under m.opt.graph_conditional) must see no sibling leaves when it is recorded.

Tasks never nest: the caller opens the `pre` task, calls the solver step with the shim
installed, and the shim exits/enters tasks through a `scopes` object with
  exit()            close the open task
  sibling(name)     context manager: open a sibling task, close it
  enter(name)       open the join task and leave it open (the caller closes it).
"""

import contextlib


class ForkScopes:
    """Sequentially opened task scopes from a dict name -> (deps...). `task(*deps, symbol=)`
    returns a context manager running the body on the task's stream (the bench recipe)."""

    def __init__(self, task, specs):
        self._task, self._specs, self._cur = task, specs, None

    def enter(self, name):
        assert self._cur is None, "a task is still open"
        self._cur = self._task(*self._specs[name], symbol=name)
        self._cur.__enter__()

    def exit(self):
        self._cur.__exit__(None, None, None)
        self._cur = None

    @contextlib.contextmanager
    def sibling(self, name):
        self.enter(name)
        try:
            yield
        finally:
            self.exit()


class MjwForkShim:
    """Module stand-in; set as `solver._mujoco_warp` (instance attribute shadowing the class
    attribute) while recording, delete afterwards. `join_name` is the scope opened last."""

    def __init__(self, mjw, scopes, join_name="join"):
        self._mjw, self._scopes, self._join = mjw, scopes, join_name
        self.steps = 0

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
        # --- pre task (open on entry)
        mjw.fwd_kinematics(m, d)
        self._scopes.exit()
        # --- siblings
        with self._scopes.sibling("con"):
            if m.opt.run_collision_detection:
                mjw.collision(m, d)
            constraint.make_constraint(m, d)
        with self._scopes.sibling("mass"):
            smooth.crb(m, d)
            smooth.tendon_armature(m, d)
        with self._scopes.sibling("vel"):
            smooth.transmission(m, d)
            mjw.fwd_velocity(m, d)
            mjw.fwd_actuation(m, d)
        # --- join
        self._scopes.enter(self._join)
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


@contextlib.contextmanager
def installed(solver, shim):
    assert not getattr(solver, "_use_speculative_contacts", False), "speculative contacts path not forked"
    solver._mujoco_warp = shim
    try:
        yield shim
    finally:
        del solver._mujoco_warp
