"""A pytest plugin that plants one wrong certificate, to show whether the misspecification tests
catch it.

``CHC_MUTANT="<kind>:<factor>"`` names the mutant; ``-p chc_mutants``, with this folder on
``PYTHONPATH``, loads the plugin, which plants it before the test modules import the library's
names. Each mutant moves one number a misspecification test reads, the way that makes the
certificate wrong (narrower, smaller, more significant), and nothing else:

    tube:k      the plan's uncertainty tube, times k                      test_misspec_tube
    channel:k   fit_causal_residual's channel_error, times k               test_misspec_channel
    ope:k       evaluate_plan's interval about its centre, times k         test_misspec_ope
    regret:k    plan_regret_bound's bound, times k                         test_misspec_regret
    dither:k    the dither check's p-value, times k                        test_misspec_dither
    lag:k       lag_check's p-value, times k                               test_misspec_lag
    lift:k      LiftFit.interval about its centre, times k                 test_misspec_lift
    gate:k      GateConfig.alpha, times k                                  test_misspec_gate
    barrier:1   every step of a plan certified by the barrier              test_misspec_barrier

``just mutants`` plants each on the file that reads it over five seed blocks; a file that fails on
four of them or more caught it. Not a test module: pytest collects nothing here.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from collections.abc import Callable
from typing import Any


def _wrap_prescribe(mutate: Callable[[Any], Any]) -> None:
    import chc
    import chc.decision as decision

    original = decision.prescribe

    @functools.wraps(original)
    def mutated(*args: Any, **kwargs: Any) -> Any:
        return mutate(original(*args, **kwargs))

    decision.prescribe = mutated
    chc.prescribe = mutated


def _wrap(module: Any, name: str, mutate: Callable[[Any], Any]) -> None:
    """``module.name`` replaced by a function that returns ``mutate`` of what it returned."""
    original = getattr(module, name)

    @functools.wraps(original)
    def mutated(*args: Any, **kwargs: Any) -> Any:
        return mutate(original(*args, **kwargs))

    setattr(module, name, mutated)


def _about_centre(interval: tuple[float, float], k: float) -> tuple[float, float]:
    low, high = interval
    centre, half = 0.5 * (low + high), 0.5 * (high - low)
    return centre - k * half, centre + k * half


def _tube(k: float) -> None:
    def narrower(result: Any) -> Any:
        if result.plan is None or result.plan.uncertainty_tube is None:
            return result
        tube = result.plan.uncertainty_tube * k
        return dataclasses.replace(
            result, plan=dataclasses.replace(result.plan, uncertainty_tube=tube)
        )

    _wrap_prescribe(narrower)


def _dither(k: float) -> None:
    def smaller(result: Any) -> Any:
        check = result.dither_check
        if check is None or check.contrast is None:
            return result
        contrast = dataclasses.replace(check.contrast, p_value=min(1.0, check.contrast.p_value * k))
        return dataclasses.replace(
            result, dither_check=dataclasses.replace(check, contrast=contrast)
        )

    _wrap_prescribe(smaller)


def _barrier(_: float) -> None:
    import chc.plan as plan

    def everything(result: Any) -> Any:
        certificate = result.certificate
        if certificate.barrier_certified_steps is None or result.plan is None:
            return result
        steps = len(result.plan.actions)
        certificate = dataclasses.replace(certificate, barrier_certified_steps=steps)
        return dataclasses.replace(result, certificate=certificate)

    _wrap_prescribe(everything)
    original = plan.certify_safety

    @functools.wraps(original)
    def certify(held: Any, *args: Any, **kwargs: Any) -> Any:
        audit = original(held, *args, **kwargs)
        return dataclasses.replace(audit, certified_steps=len(held.actions))

    plan.certify_safety = certify


def _channel(k: float) -> None:
    import chc.dynamics_id as dynamics_id

    def smaller(fit: Any) -> Any:
        if fit.channel_error is None:
            return fit
        return dataclasses.replace(fit, channel_error=fit.channel_error * k)

    _wrap(dynamics_id, "fit_causal_residual", smaller)


def _lag(k: float) -> None:
    import chc.dynamics_id as dynamics_id

    _wrap(
        dynamics_id,
        "lag_check",
        lambda found: dataclasses.replace(found, p_value=min(1.0, found.p_value * k)),
    )


def _ope(k: float) -> None:
    import chc.evaluation as evaluation

    _wrap(
        evaluation,
        "evaluate_plan",
        lambda found: dataclasses.replace(found, interval=_about_centre(found.interval, k)),
    )


def _regret(k: float) -> None:
    import chc.plan as plan

    _wrap(
        plan, "plan_regret_bound", lambda found: dataclasses.replace(found, bound=found.bound * k)
    )


def _lift(k: float) -> None:
    import chc.lift as lift

    original = lift.LiftFit.interval

    def interval(self: Any, parameter: Any) -> tuple[float, float]:
        return _about_centre(original(self, parameter), k)

    lift.LiftFit.interval = interval


def _gate(k: float) -> None:
    import chc.gate as gate

    original = gate.GateConfig

    def config(*args: Any, **kwargs: Any) -> Any:
        made = original(*args, **kwargs)
        return dataclasses.replace(made, alpha=min(0.99, made.alpha * k))

    gate.GateConfig = config


MUTANTS: dict[str, Callable[[float], None]] = {
    "tube": _tube,
    "channel": _channel,
    "ope": _ope,
    "regret": _regret,
    "dither": _dither,
    "lag": _lag,
    "lift": _lift,
    "gate": _gate,
    "barrier": _barrier,
}


def pytest_configure(config: Any) -> None:
    del config
    spec = os.environ.get("CHC_MUTANT")
    if not spec:
        return
    kind, _, factor = spec.partition(":")
    if kind not in MUTANTS:
        raise ValueError(f"no mutant {kind!r}: {__doc__}")
    MUTANTS[kind](float(factor or 1.0))
    print(f"[chc_mutants] planted {kind}:{factor or 1}")
