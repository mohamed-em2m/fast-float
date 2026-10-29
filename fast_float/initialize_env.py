import asyncio
import contextlib
import glob
import json
import os
import site

try:
    import uvloop

    # Set the event loop policy to uvloop on non-Windows platforms
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("Using uvloop (C-accelerated event loop) for maximum event loop performance.")
except (ImportError, AttributeError):
    # uvloop isn't installed or not supported on Windows
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def set_env_variables(path: str):
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as file:
        j = json.load(file)
        for var, val in j.items():
            if var.startswith("_"):
                continue  # comment / documentation keys, not env vars
            if var not in os.environ or not os.environ[var]:
                os.environ[var] = str(val)
    return True


temp_dir = os.path.join(PROJECT_ROOT, "temp")
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)


def initialize_environment():
    """Initialize environment variables and API setup"""
    candidates = [
        os.path.join(PROJECT_ROOT, "config", "api.json"),
        os.path.join(os.path.dirname(__file__), "config", "api.json"),
        "config/api.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                set_env_variables(path)
                return True
            except Exception as e:
                print(f"Failed loading config from {path}: {e}")
    return False


_SYSTEM_CUDA_LIB_DIRS = ("/usr/local/cuda/lib64", "/usr/lib64-nvidia")
_SHARED_LIB_EXTS = (".so", ".dll")


def _is_shared_lib(name: str) -> bool:
    return name.endswith(_SHARED_LIB_EXTS) or ".so." in name


def _dedupe(items):
    """Order-preserving dedupe."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _discover_gpu_lib_dirs(package_paths):
    """Directories under pip ``nvidia``/``tensorrt*`` packages that hold shared libs.

    ORT's CUDA EP dlopens ``libcublas``/``libcudnn`` and the TensorRT EP
    dlopens ``libnvinfer`` by soname; when those ship as pip wheels
    (``nvidia-*``, ``tensorrt``, ``tensorrt_libs``, ``tensorrt_cu12_libs``)
    they are not on the default loader path, so ORT silently degrades the
    session to CPU (discarding the CUDA EP too).
    """
    lib_dirs = []
    for base_path in package_paths:
        roots = [os.path.join(base_path, name) for name in ("nvidia", "tensorrt", "tensorrt_libs")]
        roots += glob.glob(os.path.join(base_path, "tensorrt*"))
        for root_dir in roots:
            if not os.path.isdir(root_dir):
                continue
            for root, _dirs, files in os.walk(root_dir):
                if any(_is_shared_lib(f) for f in files):
                    lib_dirs.append(root)
    lib_dirs.extend(path for path in _SYSTEM_CUDA_LIB_DIRS if os.path.isdir(path))
    return _dedupe(lib_dirs)


def initialize_onnx_lib_paths():
    """Put pip-provided CUDA / TensorRT libraries on the loader path.

    Without this, ``onnxruntime``'s CUDA EP cannot resolve ``libcublas`` /
    ``libcudnn`` and the TensorRT EP cannot resolve ``libnvinfer``, so ORT
    drops every GPU provider and silently runs on CPU.
    """
    package_paths = site.getsitepackages()
    if hasattr(site, "getusersitepackages"):
        package_paths.append(site.getusersitepackages())

    lib_dirs = _discover_gpu_lib_dirs(package_paths)

    if os.name == "nt":
        for path in lib_dirs:
            with contextlib.suppress(AttributeError, OSError):
                os.add_dll_directory(path)
        os.environ["PATH"] = os.pathsep.join([*lib_dirs, os.environ.get("PATH", "")])
    else:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs) + (os.pathsep + current_ld if current_ld else "")

    print("Discovered and added CUDA/NVIDIA/TensorRT library paths:")
    for path in lib_dirs:
        print(f" - {path}")

    # Quick validation inside Python
    try:
        import onnxruntime as ort

        print("\nAvailable Execution Providers:", ort.get_available_providers())
    except Exception as e:
        print("\nFailed to test ONNX Runtime:", e)
