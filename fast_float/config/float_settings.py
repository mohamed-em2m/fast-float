"""FLOAT model/inference options loaded from ``settings.toml``.

``[float]`` is the operator's single source of truth for the FLOAT pipeline:
checkpoint paths, model architecture, audio/video windows, ODE sampling,
classifier-free guidance, diffusion/ablation settings, decode strategy and the
defaults the HTTP API uses when a request omits a field.

The table is intentionally nested for readability. Every leaf in the model
tables is applied to the matching
:class:`~fast_float.core.options.InferenceOptions` attribute (the argparse
destination), so ``[float.ode].nfe`` sets ``opt.nfe``. Nested table names are
only documentation; the loader flattens them. Two subtables are special:
``[float.request]`` feeds :func:`request_default` (the HTTP schema defaults) and
``[float].livekit_nfe`` is read by the LiveKit renderer.

Precedence (highest first):

1. ``settings.toml`` ``[float]`` (the shared operator config)
2. the legacy environment variables (``NGPUS``, ``OPTIMIZATION_MODE``, ...)
3. the argparse / built-in defaults

That order keeps one file authoritative for the API and the LiveKit worker,
while boxes without the shared file keep working through the old env vars.

.. warning::

   Architecture values (``dim_*``, ``fmt_depth``, ``input_size``, ``fps``,
   ``wav2vec_sec``, ...) are part of the checkpoint's shape. They must match
   ``checkpoints/float.pth``; changing them makes ``load_state_dict`` fail or
   silently produces the wrong motion. Adjust the sampling/decode/guidance
   knobs instead.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterator
from typing import Any

from fast_float.config.dynamic import load_settings

logger = logging.getLogger(__name__)

FLOAT_SECTION = "float"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

#: Opt attributes that are read with ``getattr(opt, ...)`` but have no CLI flag.
_EXTRA_OPTIONS: dict[str, tuple[type, Any]] = {
    "decode_batch_size": (int, None),
    "decode_channels_last": (bool, None),
    "decode_autocast": (str, None),
}

#: Legacy environment fallbacks, used only when ``[float]`` omits the key.
_ENV_FALLBACKS: dict[str, tuple[str, type, Any]] = {
    "ngpus": ("NGPUS", int, None),
    "optimization_mode": ("OPTIMIZATION_MODE", str, None),
    "use_fp16": ("USE_FP16", bool, True),
    "res_dir": ("RESULTS_DIR", str, None),
    "decode_channels_last": ("FLOAT_DECODE_CHANNELS_LAST", bool, None),
    "decode_autocast": ("FLOAT_DECODE_AUTOCAST", str, None),
}

#: ``[float]`` subtables that are not model options.
_NON_OPT_TABLES = {"livekit", "request"}
#: Top-level ``[float]`` keys that are not model options.
_NON_OPT_KEYS = {"livekit_nfe"}

#: Defaults for :class:`~fast_float.schemas.generation.GenerationRequest`
#: fields: ``name -> (default, type, minimum, maximum, choices)``.
_REQUEST_DEFAULTS: dict[str, tuple[Any, type, float | None, float | None, tuple[str, ...] | None]] = {
    "nfe": (10, int, 5, 25, None),
    "seed": (25, int, 0, None, None),
    "emotion": ("S2E", str, None, None, ("S2E", "neutral", "happy", "sad", "angry", "surprise", "disgust")),
    "a_cfg_scale": (2.0, float, 0.5, 5.0, None),
    "r_cfg_scale": (1.0, float, 0.5, 3.0, None),
    "e_cfg_scale": (1.0, float, 0.5, 3.0, None),
    "no_crop": (False, bool, None, None, None),
    "decode_batch_size": (0, int, 0, 64, None),
}

_INVALID = object()


def float_table() -> dict[str, Any]:
    """The effective ``[float]`` table (``{}`` when absent or malformed)."""
    table = load_settings().get(FLOAT_SECTION, {})
    if not isinstance(table, dict):
        logger.warning("settings.toml [%s] must be a table; ignoring %r", FLOAT_SECTION, type(table).__name__)
        return {}
    return table


def _flatten(table: dict[str, Any], prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield ``(path, value)`` for every non-table leaf, depth first."""
    for key, value in table.items():
        path = (*prefix, str(key))
        if isinstance(value, dict):
            yield from _flatten(value, path)
        else:
            yield path, value


def _as_bool(value: Any) -> bool | object:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return _INVALID


def _coerce(
    raw: Any,
    kind: type,
    *,
    choices: tuple[str, ...] | None = None,
    default: Any = None,
) -> Any:
    """Coerce a TOML/env value to ``kind``; return :data:`_INVALID` on error."""
    if kind is bool:
        return _as_bool(raw)
    if kind is str:
        text = str(raw).strip()
        if not text:
            return None if default is None else _INVALID
        if choices and text not in choices:
            return _INVALID
        return text
    text = str(raw).strip()
    try:
        if kind is int:
            return int(text)
        if kind is float:
            return float(text)
    except (TypeError, ValueError):
        return _INVALID
    return text


def _is_flag(action: argparse.Action) -> bool:
    """True for ``store_true`` / ``store_false`` (a TOML bool, not a string)."""
    return action.nargs == 0 and isinstance(action.const, bool)


def _parser() -> argparse.ArgumentParser:
    """Build the FLOAT options parser without touching ``sys.argv``."""
    from fast_float.core.options import InferenceOptions

    parser = argparse.ArgumentParser(add_help=False)
    return InferenceOptions().initialize(parser)


def _coerce_for(
    dest: str,
    raw: Any,
    action: argparse.Action | None,
) -> Any:
    if action is not None:
        kind = bool if _is_flag(action) else action.type or str
        value = _coerce(raw, kind, choices=action.choices, default=action.default)
    else:
        kind, default = _EXTRA_OPTIONS[dest]
        value = _coerce(raw, kind, default=default)
        if dest == "decode_autocast" and value is None:
            # An explicitly empty value means "off" and must not silently fall
            # back to the legacy FLOAT_DECODE_AUTOCAST env var.
            value = "off"
    return value


def _apply_env_fallbacks(opt: argparse.Namespace, seen: set[str]) -> None:
    for dest, (env_name, kind, fallback) in _ENV_FALLBACKS.items():
        if dest in seen:
            continue
        raw = os.getenv(env_name, "").strip()
        if not raw:
            if fallback is not None:
                setattr(opt, dest, fallback)
            continue
        value = _coerce(raw, kind, default=getattr(opt, dest, fallback))
        if value is _INVALID:
            logger.warning("ignoring invalid %s=%r (settings.toml [float] is unset)", env_name, raw)
            continue
        setattr(opt, dest, value)


def load_float_options() -> argparse.Namespace:
    """Build the FLOAT options namespace: settings.toml, then env, then defaults.

    Returns the same object shape as ``InferenceOptions().parse()`` plus the
    getattr-only decode knobs. ``main`` applies it to ``app_state.opt`` before
    constructing :class:`OptimizedInferenceAgent`.
    """
    parser = _parser()
    opt = parser.parse_args([])
    actions = {action.dest: action for action in parser._actions}
    seen: set[str] = set()
    unknown: list[str] = []

    for path, raw in _flatten(float_table()):
        dest = path[-1]
        if path[0] in _NON_OPT_TABLES or (len(path) == 1 and dest in _NON_OPT_KEYS):
            continue
        if dest not in actions and dest not in _EXTRA_OPTIONS:
            unknown.append(".".join(path))
            continue

        value = _coerce_for(dest, raw, actions.get(dest))
        if value is _INVALID:
            fallback = getattr(opt, dest, _EXTRA_OPTIONS.get(dest, (str, None))[1])
            logger.warning(
                "settings.toml [float].%s=%r is invalid; keeping %r",
                ".".join(path),
                raw,
                fallback,
            )
            continue
        setattr(opt, dest, value)
        seen.add(dest)

    # Materialize the getattr-only knobs so downstream reads are deterministic.
    for dest, (_, default) in _EXTRA_OPTIONS.items():
        if not hasattr(opt, dest):
            setattr(opt, dest, default)

    _apply_env_fallbacks(opt, seen)

    if unknown:
        logger.warning(
            "settings.toml [float] has unknown keys (ignored): %s",
            ", ".join(sorted(unknown)),
        )
    return opt


def request_default(name: str) -> Any:
    """A ``[float.request]`` default for the HTTP generation schema.

    Request fields sent by the client always win; these values are the fallback
    for omitted fields. Values are coerced and clamped to the schema's range so
    an operator edit can never make ``GenerationRequest`` reject its own
    default.
    """
    try:
        default, kind, minimum, maximum, choices = _REQUEST_DEFAULTS[name]
    except KeyError as exc:  # pragma: no cover - programming error
        raise KeyError(f"no FLOAT request default named {name!r}") from exc

    table = float_table().get("request", {})
    raw = table.get(name, default) if isinstance(table, dict) else default
    value = _coerce(raw, kind, choices=choices, default=default)
    if value is _INVALID or value is None:
        logger.warning("settings.toml [float.request].%s=%r is invalid; keeping %r", name, raw, default)
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


__all__ = [
    "FLOAT_SECTION",
    "float_table",
    "load_float_options",
    "request_default",
]
