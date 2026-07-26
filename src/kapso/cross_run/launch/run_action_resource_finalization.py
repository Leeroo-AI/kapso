"""Sealed controller authority between durable terminals and physical resources."""

from __future__ import annotations

import os
from threading import Lock
from typing import Protocol
from weakref import WeakKeyDictionary

from kapso.cross_run.launch.run_action_store import RunActionExecutionStore
from kapso.cross_run.settings import LaunchSettings

_RESOURCE_FINALIZATION_AUTHORITY_ISSUANCE = object()
_RESOURCE_FINALIZATION_LOCK = Lock()
_RESOURCE_FINALIZATION_BINDINGS: WeakKeyDictionary[
    RunActionResourceFinalizationAuthority, tuple
] = WeakKeyDictionary()
_DRIVER_METHOD_NAMES = (
    "finalize_terminal",
    "require_terminal_absence",
)


class RunActionResourceFinalizationError(RuntimeError):
    """Terminal resource finalization authority is absent or incompatible."""


class RunActionResourceFinalizationDriver(Protocol):
    """Physical resource driver retained behind the sealed controller authority."""

    def finalize_terminal(self, operation_id: str) -> None: ...

    def require_terminal_absence(self, operation_id: str) -> None: ...


class RunActionResourceFinalizationAuthority:
    """Process-bound authority permanently joined to one store, settings, and driver."""

    def __init__(
        self,
        *,
        action_store: RunActionExecutionStore,
        launch_settings: LaunchSettings,
        driver: RunActionResourceFinalizationDriver,
        _authority: object,
    ) -> None:
        methods = tuple(
            getattr(type(driver), name, None) for name in _DRIVER_METHOD_NAMES
        )
        if (
            type(action_store) is not RunActionExecutionStore
            or type(launch_settings) is not LaunchSettings
            or _authority is not _RESOURCE_FINALIZATION_AUTHORITY_ISSUANCE
            or any(method is None for method in methods)
            or any(
                getattr(getattr(driver, name), "__self__", None) is not driver
                or getattr(getattr(driver, name), "__func__", None) is not method
                for name, method in zip(_DRIVER_METHOD_NAMES, methods)
            )
        ):
            raise RunActionResourceFinalizationError(
                "run-action resource finalization authority lacks exact issuance"
            )
        self._owner_process_id = os.getpid()
        binding = (
            action_store,
            launch_settings,
            driver,
            type(driver),
            methods,
        )
        with _RESOURCE_FINALIZATION_LOCK:
            if _RESOURCE_FINALIZATION_BINDINGS.get(self) is not None:
                raise RunActionResourceFinalizationError(
                    "run-action resource finalization authority is already issued"
                )
            _RESOURCE_FINALIZATION_BINDINGS[self] = binding

    def finalize_terminal(self, operation_id: str) -> None:
        """Finalize one durable terminal before it can enter a recovery report."""

        _require_operation_id(operation_id)
        driver = self._require_current()[2]
        result = driver.finalize_terminal(operation_id)
        if result is not None:
            raise RunActionResourceFinalizationError(
                "resource finalization driver returned unauthorized data"
            )

    def require_terminal_absence(self, operation_id: str) -> None:
        """Reprove one terminal resource set before later durable transitions."""

        _require_operation_id(operation_id)
        driver = self._require_current()[2]
        result = driver.require_terminal_absence(operation_id)
        if result is not None:
            raise RunActionResourceFinalizationError(
                "resource absence driver returned unauthorized data"
            )

    def _require_binding(
        self,
        action_store: RunActionExecutionStore,
        launch_settings: LaunchSettings,
    ) -> None:
        binding = self._require_current()
        if binding[0] is not action_store or binding[1] is not launch_settings:
            raise RunActionResourceFinalizationError(
                "run-action resource finalization authority differs from its run"
            )

    def _require_current(self) -> tuple:
        with _RESOURCE_FINALIZATION_LOCK:
            binding = _RESOURCE_FINALIZATION_BINDINGS.get(self)
        if (
            type(self) is not RunActionResourceFinalizationAuthority
            or self._owner_process_id != os.getpid()
            or type(binding) is not tuple
            or len(binding) != 5
            or type(binding[0]) is not RunActionExecutionStore
            or type(binding[1]) is not LaunchSettings
            or type(binding[2]) is not binding[3]
            or any(
                getattr(getattr(binding[2], name), "__self__", None) is not binding[2]
                or getattr(getattr(binding[2], name), "__func__", None) is not method
                for name, method in zip(_DRIVER_METHOD_NAMES, binding[4])
            )
        ):
            raise RunActionResourceFinalizationError(
                "run-action resource finalization authority is foreign or changed"
            )
        return binding


def _issue_run_action_resource_finalization_authority(
    *,
    action_store: RunActionExecutionStore,
    launch_settings: LaunchSettings,
    driver: RunActionResourceFinalizationDriver,
) -> RunActionResourceFinalizationAuthority:
    return RunActionResourceFinalizationAuthority(
        action_store=action_store,
        launch_settings=launch_settings,
        driver=driver,
        _authority=_RESOURCE_FINALIZATION_AUTHORITY_ISSUANCE,
    )


def require_run_action_resource_finalization_authority(
    authority: RunActionResourceFinalizationAuthority,
    action_store: RunActionExecutionStore,
    launch_settings: LaunchSettings,
) -> None:
    """Require one current authority for the publisher's exact run."""

    if type(authority) is not RunActionResourceFinalizationAuthority:
        raise RunActionResourceFinalizationError(
            "run-action resource finalization authority has an invalid type"
        )
    authority._require_binding(action_store, launch_settings)


def _require_operation_id(operation_id: str) -> None:
    if type(operation_id) is not str or not operation_id:
        raise RunActionResourceFinalizationError(
            "resource finalization requires one operation identity"
        )


__all__ = [
    "require_run_action_resource_finalization_authority",
    "RunActionResourceFinalizationAuthority",
    "RunActionResourceFinalizationError",
]
