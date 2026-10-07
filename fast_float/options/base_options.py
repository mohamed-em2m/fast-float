"""
Compatibility shim for legacy `options` imports.
Redirects to `fast_float.schemas.base_options`.
"""
try:
    from fast_float.schemas.base_options import BaseOptions, save_options, load_options
except ImportError:
    from schemas.base_options import BaseOptions, save_options, load_options

__all__ = ["BaseOptions", "save_options", "load_options"]
