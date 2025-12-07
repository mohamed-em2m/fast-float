import os 
import json
import asyncio 
import uvloop
try:
    
    # Set the event loop policy to uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("Using uvloop (C-accelerated event loop) for maximum event loop performance.")
except ImportError:
    print("uvloop not installed. Using default CPython event loop (which is also highly optimized).")
    pass # If uvloop isn't installed, we stick with the default

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def set_env_variables(path):
    with open(f"{path}", "r") as file:
        j = json.load(file)
        for var in j.keys():
            os.environ[var] = str(j.get(var))  

if not os.path.exists( "temp"):
    os.makedirs( "temp")


def initialize_environment():
    """Initialize environment variables and API setup"""
    try:
        set_env_variables("config/api.json")
        return True
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return False    
    
