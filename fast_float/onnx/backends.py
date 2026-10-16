"""Config-driven per-component backend selection.

FLOAT's inference pipeline is split into named *components*, each of which can
run in one of two accelerated modes -- ``onnx`` (the exported graph on an
execution provider, TensorRT preferred; ``provider`` records which) or
``compile`` (``torch.compile``) -- with plain torch (``eager``) as the
always-available baseline and fallback.

The choice is *hardware-specific* (TensorRT/CUDA availability, GPU model), so
the config is a build artifact produced on the target box by the selector and
read here at load time. When the config is missing or unreadable every
component falls back to ``eager`` -- i.e. today's behaviour, unchanged.

Resolution order (first hit wins):
  1. explicit ``path`` argument
  2. ``$FLOAT_BACKENDS_CONFIG``
  3. ``<repo>/onnx/backends.generated.yaml``  (gitignored, target-specific)
  4. ``<repo>/config/backends.yaml``          (committed safe default)
  5. built-in default: every component ``eager``

Per-component override for debugging: ``FLOAT_MODE_<COMPONENT>`` (e.g.
``FLOAT_MODE_fmt=eager``).

See ``docs/BACKEND_SELECTION.md`` for the full design.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COMPONENTS = ("emotion", "audio", "direction", "fmt", "decoder", "refenc")
HOT_COMPONENTS = ("fmt", "decoder", "audio")
MODES = ("eager", "compile", "onnx")
ONNX_PROVIDERS = ("trt", "cuda", "cpu")
DEFAULT_MODE = "eager"

ENV_CONFIG = "FLOAT_BACKENDS_CONFIG"
ENV_MODE_PREFIX = "FLOAT_MODE_"

_REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_CONFIG = _REPO_ROOT / "onnx" / "backends.generated.yaml"
DEFAULT_CONFIG = _REPO_ROOT / "config" / "backends.yaml"


@dataclass
class ComponentConfig:
    """Resolved choice for one pipeline component."""

    name: str
    mode: str = DEFAULT_MODE
    provider: str | None = None
    onnx_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    @property
    def accelerated(self) -> bool:
        return self.mode != DEFAULT_MODE


@dataclass
class BackendConfig:
    """Whole-pipeline backend config (the parsed YAML)."""

    version: int = 2
    source: str = "<default>"
    target: dict[str, Any] = field(default_factory=dict)
    objective: str = "end_to_end_ms"
    default_mode: str = DEFAULT_MODE
    chosen_combo: dict[str, str] = field(default_factory=dict)
    e2e: dict[str, Any] = field(default_factory=dict)
    components: dict[str, ComponentConfig] = field(default_factory=dict)

    def component(self, name: str) -> ComponentConfig:
        return self.components.get(name) or ComponentConfig(name=name, mode=self.default_mode)

    def mode_for(self, name: str) -> str:
        override = os.environ.get(ENV_MODE_PREFIX + name.upper())
        if override in MODES:
            return override
        return self.component(name).mode

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target": dict(self.target),
            "objective": self.objective,
            "default_mode": self.default_mode,
            "chosen_combo": dict(self.chosen_combo),
            "e2e": dict(self.e2e),
            "components": {
                name: {
                    "mode": c.mode,
                    **({"provider": c.provider} if c.provider else {}),
                    **({"onnx_path": c.onnx_path} if c.onnx_path else {}),
                    **({"metrics": c.metrics} if c.metrics else {}),
                    **({"reason": c.reason} if c.reason else {}),
                }
                for name, c in self.components.items()
            },
        }


def default_config() -> BackendConfig:
    """All-eager config covering every known component."""
    return BackendConfig(
        source="<default>",
        components={name: ComponentConfig(name=name) for name in COMPONENTS},
    )


def _coerce_mode(mode: Any, name: str) -> str:
    mode = str(mode)
    if mode not in MODES:
        print(f"[backends] WARNING: unknown mode {mode!r} for {name!r}; using eager")
        return DEFAULT_MODE
    return mode


def _coerce_provider(provider: Any) -> str | None:
    if provider is None:
        return None
    provider = str(provider)
    if provider not in ONNX_PROVIDERS:
        print(f"[backends] WARNING: unknown onnx provider {provider!r}; ignoring")
        return None
    return provider


def _from_dict(data: dict[str, Any], source: str = "<dict>") -> BackendConfig:
    default_mode = _coerce_mode(data.get("default_mode", DEFAULT_MODE), "default_mode")
    components: dict[str, ComponentConfig] = {}
    for name, cd in (data.get("components") or {}).items():
        cd = cd or {}
        components[name] = ComponentConfig(
            name=name,
            mode=_coerce_mode(cd.get("mode", default_mode), name),
            provider=_coerce_provider(cd.get("provider")),
            onnx_path=cd.get("onnx_path"),
            metrics=dict(cd.get("metrics") or {}),
            reason=cd.get("reason"),
        )
    # Every known component is present: unspecified ones default to eager, so
    # a partial/generated config is self-describing at load time.
    for name in COMPONENTS:
        components.setdefault(name, ComponentConfig(name=name, mode=default_mode, reason="unspecified"))
    return BackendConfig(
        version=int(data.get("version", 2)),
        source=source,
        target=dict(data.get("target") or {}),
        objective=str(data.get("objective", "end_to_end_ms")),
        default_mode=default_mode,
        chosen_combo=dict(data.get("chosen_combo") or {}),
        e2e=dict(data.get("e2e") or {}),
        components=components,
    )


def _from_file(path: Path) -> BackendConfig:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping at top level, got {type(data).__name__}")
    return _from_dict(data, source=str(path))


def load_backends(path: str | None = None) -> BackendConfig:
    """Load the active backend config (never raises for the implicit search).

    An explicit ``path`` that does not exist raises ``FileNotFoundError`` --
    asking for a specific file that is missing is a caller error. Implicit
    candidates that are missing or unreadable are skipped and we fall through
    to the safe all-eager default.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"backends config not found: {p}")
        return _from_file(p)

    candidates = []
    env_path = os.environ.get(ENV_CONFIG)
    if env_path:
        candidates.append(Path(env_path))
    candidates += [GENERATED_CONFIG, DEFAULT_CONFIG]
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            return _from_file(cand)
        except Exception as e:
            print(f"[backends] WARNING: ignoring unreadable config {cand}: {e}")
    return default_config()


def save_backends(cfg: BackendConfig, path: str | Path) -> Path:
    """Write ``cfg`` to ``path`` as YAML (used by the selector)."""
    import yaml

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(cfg.as_dict(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return p


def cuda_ep_usable() -> bool:
    """True only when the CUDA EP actually initializes on a real tiny session.

    A listed EP can still fail to load -- e.g. torch ``cu130`` with an ORT
    ``cu12`` wheel makes ORT drop every GPU provider (``libcublasLt.so.12``
    missing) and silently run on CPU. Session creation is the only honest test.
    """
    try:
        import onnxruntime as ort
        from onnx import TensorProto, helper

        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "ep_probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        active = sess.get_providers()
        return bool(active) and active[0] != "CPUExecutionProvider"
    except Exception:
        return False


_CACHED_CONFIG: BackendConfig | None = None


def get_config(reload: bool = False) -> BackendConfig:
    """Process-wide cached config (env overrides are applied at lookup time)."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None or reload:
        _CACHED_CONFIG = load_backends()
    return _CACHED_CONFIG


def resolve_mode(name: str) -> str:
    """Active mode for ``name`` (config + ``FLOAT_MODE_<NAME>`` override)."""
    return get_config().mode_for(name)


def resolve_provider(name: str) -> str | None:
    return get_config().component(name).provider


def maybe_compile(module, name: str):
    """Return ``torch.compile(module)`` when the config selects ``compile``.

    Behaviour-preserving; falls back to the eager module when compile is
    unavailable or raises at wrap time.
    """
    if resolve_mode(name) != "compile":
        return module
    try:
        import torch

        if not hasattr(torch, "compile"):
            print(f"[backends:{name}] torch.compile unavailable; eager")
            return module
        print(f"[backends:{name}] torch.compile enabled")
        return torch.compile(module)
    except Exception as e:
        print(f"[backends:{name}] torch.compile failed ({e}); eager")
        return module


def env_fingerprint() -> dict[str, Any]:
    """Best-effort hardware/runtime probe; never raises.

    ``cuda_ep`` reflects an actual session test (see :func:`cuda_ep_usable`),
    not just whether ORT lists the provider.
    """
    info: dict[str, Any] = {}
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            try:
                props = torch.cuda.get_device_properties(0)
                info["gpu"] = str(props.name)
                info["cc"] = f"{props.major}.{props.minor}"
                info["vram_gb"] = round(props.total_memory / (1 << 30), 1)
            except Exception:
                pass
    except Exception as e:
        info["torch"] = f"<unavailable: {e}>"
        info["cuda"] = False

    try:
        import onnxruntime as ort

        info["ort"] = str(ort.__version__)
        available = set(ort.get_available_providers())
        info["trt_ep_listed"] = "TensorrtExecutionProvider" in available
    except Exception as e:
        info["ort"] = f"<unavailable: {e}>"
        info["trt_ep_listed"] = False

    try:
        from fast_float.onnx.export import _trt_libs_loadable

        info["trt_libs_loadable"] = bool(_trt_libs_loadable())
    except Exception:
        info["trt_libs_loadable"] = False

    info["trt"] = bool(info.get("trt_ep_listed") and info.get("trt_libs_loadable"))
    info["cuda_ep"] = cuda_ep_usable() if info.get("cuda") else False
    return info


__all__ = [
    "COMPONENTS",
    "DEFAULT_CONFIG",
    "DEFAULT_MODE",
    "GENERATED_CONFIG",
    "HOT_COMPONENTS",
    "MODES",
    "ONNX_PROVIDERS",
    "BackendConfig",
    "ComponentConfig",
    "cuda_ep_usable",
    "default_config",
    "env_fingerprint",
    "get_config",
    "load_backends",
    "maybe_compile",
    "resolve_mode",
    "resolve_provider",
    "save_backends",
]
