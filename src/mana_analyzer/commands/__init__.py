"""Command submodule exports for mana_analyzer CLI.

Import side-effects to register optional Typer commands.
"""

# Optional command module: do not crash if a build omits this file.
try:
    from .show_flow_cmd import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass

