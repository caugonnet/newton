# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gateway helpers for the experimental ``warp.stf_experimental`` module.

The CUDASTF (CUDA Stream Tasking Framework) integration in Warp is exposed
through ``warp.stf_experimental``. That module is optional: it is only
available in Warp builds that ship the experimental task-graph API and only
usable on CUDA devices where ``wp_stf.is_available()`` returns True.

This module isolates the import and availability check so the rest of
Newton can opt into STF parallelism without scattering try/except blocks
or repeated availability probes across solvers.

Solvers that want to expose an STF-accelerated path typically:

* gate the path behind an explicit user flag (constructor arg or env var);
* call :func:`get_wp_stf` to obtain the module when the flag is on, falling
  back to the regular path when ``get_wp_stf()`` returns ``None``;
* wrap ``wp.ScopedCapture`` with :func:`stf_capture_kwargs` so the captured
  CUDA graph uses ``CaptureMode.RELAXED`` (required for STF stream tasking
  semantics inside the captured graph).

All helpers here are no-ops when STF is unavailable, so importing this
module is safe regardless of the active Warp build.
"""

from __future__ import annotations

from typing import Any

import warp as wp

__all__ = [
    "get_wp_stf",
    "stf_capture_kwargs",
    "stf_scoped_capture",
]


_wp_stf_module: Any = None
_wp_stf_checked: bool = False


def get_wp_stf() -> Any | None:
    """Return ``warp.stf_experimental`` when installed and usable, else ``None``.

    The result is cached after the first call so repeated lookups in hot
    paths are cheap. ``None`` is returned when the import fails or when
    ``wp_stf.is_available()`` reports the runtime can't use STF (e.g. no
    CUDA device, unsupported driver).
    """

    global _wp_stf_checked, _wp_stf_module

    if _wp_stf_checked:
        return _wp_stf_module

    _wp_stf_checked = True
    try:
        import warp.stf_experimental as wp_stf  # pyright: ignore[reportMissingImports]  # noqa: PLC0415
    except ImportError:
        return None

    if wp_stf.is_available():
        _wp_stf_module = wp_stf

    return _wp_stf_module


def stf_capture_kwargs(enable: bool) -> dict[str, Any]:
    """Return ``ScopedCapture`` kwargs that enable STF-compatible capture.

    When ``enable`` is ``True`` and the running Warp build exposes
    ``wp.CaptureMode.RELAXED``, the returned dict selects relaxed capture
    mode (a prerequisite for stream-tasking semantics inside the captured
    graph). Otherwise an empty dict is returned so the call site keeps
    Warp's default capture mode.
    """

    if not enable:
        return {}

    capture_mode = getattr(wp, "CaptureMode", None)
    if capture_mode is None:
        return {}

    return {"capture_mode": capture_mode.RELAXED}


def stf_scoped_capture(enable: bool, **extra: Any) -> wp.ScopedCapture:
    """Return a :class:`wp.ScopedCapture` configured for STF when requested.

    Equivalent to ``wp.ScopedCapture(**stf_capture_kwargs(enable), **extra)``.
    Extra keyword arguments are forwarded to ``ScopedCapture`` unchanged
    and override the STF-related defaults on conflict.
    """

    kwargs = stf_capture_kwargs(enable)
    kwargs.update(extra)
    return wp.ScopedCapture(**kwargs)
