from .adapter import register as register_platform
from .tools import register_tools


def register(ctx):
    register_tools(ctx)
    register_platform(ctx)

__all__ = ["register"]
