"""
Compatibility shim for legacy `options` imports.
Redirects to `fast_float.schemas.base_options`.
"""

try:
    from fast_float.schemas.base_options import BaseOptions, load_options, save_options
except ImportError:
    from schemas.base_options import BaseOptions, load_options, save_options

__all__ = ["BaseOptions", "load_options", "save_options"]
