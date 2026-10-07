import os 
import json
import asyncio 

try:
    import uvloop
    # Set the event loop policy to uvloop on non-Windows platforms
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("Using uvloop (C-accelerated event loop) for maximum event loop performance.")
except (ImportError, AttributeError):
    # uvloop isn't installed or not supported on Windows
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def set_env_variables(path: str):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as file:
        j = json.load(file)
        for var, val in j.items():
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
        "config/api.json"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                set_env_variables(path)
                return True
            except Exception as e:
                print(f"Failed loading config from {path}: {e}")
    return False
