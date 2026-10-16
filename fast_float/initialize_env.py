import asyncio
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
    pass

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


def initialize_onnx_lib_paths():
    # Get all site-packages directories
    package_paths = site.getsitepackages()
    if hasattr(site, "getusersitepackages"):
        package_paths.append(site.getusersitepackages())

    # Find all subdirectories containing .so files under nvidia/ packages
    nvidia_lib_paths = []
    for base_path in package_paths:
        nvidia_dir = os.path.join(base_path, "nvidia")
        if os.path.exists(nvidia_dir):
            for root, _dirs, files in os.walk(nvidia_dir):
                if any(f.endswith(".so") or ".so." in f for f in files):
                    nvidia_lib_paths.append(root)

    # Also check common system library paths
    system_cuda_paths = ["/usr/local/cuda/lib64", "/usr/lib64-nvidia"]
    nvidia_lib_paths.extend(path for path in system_cuda_paths if os.path.exists(path))

    # Deduplicate and update LD_LIBRARY_PATH
    nvidia_lib_paths = list(set(nvidia_lib_paths))
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = ":".join(nvidia_lib_paths) + (":" + current_ld if current_ld else "")
    os.environ["LD_LIBRARY_PATH"] = new_ld

    print("Discovered and added CUDA/NVIDIA library paths:")
    for path in nvidia_lib_paths:
        print(f" - {path}")

    # Quick validation inside Python
    try:
        import onnxruntime as ort

        print("\nAvailable Execution Providers:", ort.get_available_providers())
    except Exception as e:
        print("\nFailed to test ONNX Runtime:", e)
